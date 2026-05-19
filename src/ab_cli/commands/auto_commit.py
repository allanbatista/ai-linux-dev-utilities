#!/usr/bin/env python3
"""
auto-commit - Automatically generate commit messages using LLM.

Captures uncommitted changes and generates appropriate commit messages.
"""
import argparse
import json
import os
import re
import sys

from ab_cli.core.config import get_language
from ab_cli.core.llm_settings import add_llm_request_arguments
from ab_cli.commands.pr_description import (
    check_gh_authenticated,
    check_gh_installed,
    create_pr,
    generate_pr_content,
)
from ab_cli.utils import (
    call_llm_with_model_info,
    log_info,
    log_success,
    log_warning,
    log_error,
    RED,
    GREEN,
    YELLOW,
    BLUE,
    NC,
    is_git_repo,
    get_repo_root,
    get_current_branch,
    is_protected_branch,
    create_branch,
    detect_base_branch,
    get_commits_ahead,
    get_diff_against_base,
    get_commits_log,
    get_files_changed,
    get_staged_files,
    get_unstaged_files,
    get_untracked_files,
    get_staged_diff,
    get_staged_name_status,
    stage_all_files,
    create_commit,
    get_latest_commit,
    get_recent_commits,
    push_branch,
)


def normalize_branch_name(branch_name: str) -> str:
    """Normalize a suggested branch name."""
    branch_name = branch_name.strip('"\'`')
    branch_name = branch_name.split('\n')[0].strip()
    branch_name = re.sub(r'\s+', '-', branch_name)
    branch_name = re.sub(r'[^a-zA-Z0-9/_-]', '', branch_name)
    if len(branch_name) > 50:
        branch_name = branch_name[:50].rstrip('-')
    return branch_name


def extract_json_object(text: str) -> dict:
    """Extract a JSON object from model output."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def generate_commit_plan(
    diff: str,
    name_status: str,
    recent_commits: str,
    current_branch: str,
    lang: str,
    reasoning_effort: str = None,
    service_tier: str = None,
) -> dict:
    """Generate branch and commit metadata using a single LLM request."""
    prompt_text = f"""Analyze the git changes below and return ONLY valid JSON.

Schema:
{{
  "branch_name": "feature/add-user-authentication",
  "commit_message": "feat: add user authentication"
}}

Rules:
1. Return ONLY a JSON object, nothing else
2. Use language: {lang}
3. branch_name must use lowercase kebab-case with an appropriate prefix:
   - feature/ for new features
   - fix/ for bug fixes
   - chore/ for maintenance tasks
   - refactor/ for code refactoring
   - docs/ for documentation
   - test/ for test-related changes
4. commit_message must be concise and ready to use as git commit content
5. Include no markdown fences, explanations, or extra keys unless useful

CURRENT BRANCH:
{current_branch}

RECENT COMMITS (style reference):
{recent_commits}

FILES CHANGED:
{name_status}

DIFF:
{diff}

Return the JSON object now:
"""

    result, selected_model, estimated_tokens = call_llm_with_model_info(
        prompt_text,
        lang=lang,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
    )

    log_info(f"Estimated tokens: ~{estimated_tokens} | Model: {selected_model} | Lang: {lang}")
    print()

    if not result:
        raise RuntimeError("API call failed for commit plan generation")

    try:
        if isinstance(result, dict):
            response_text = result.get('text', '')
        else:
            response_text = str(result)
        data = extract_json_object(response_text)
    except Exception as e:
        raise RuntimeError(f"Invalid structured response: {e}") from e

    branch_name = normalize_branch_name(str(data.get('branch_name', '') or ''))
    commit_message = str(data.get('commit_message', '') or '').strip()

    return {
        'branch_name': branch_name,
        'commit_message': commit_message,
    }


def handle_protected_branch(current_branch: str, suggested_branch: str) -> str:
    """Handle protected branch workflow and return the selected branch."""
    log_warning(f"You are on '{current_branch}' branch.")
    print()

    if suggested_branch:
        print(f"\n{GREEN}Suggested branch:{NC} {YELLOW}{suggested_branch}{NC}\n")
    else:
        log_warning("Could not suggest branch name")
        print()

    print("Options:")
    if suggested_branch:
        print(f"  {GREEN}[1]{NC} Create suggested branch '{suggested_branch}'")
    else:
        print(f"  {GREEN}[1]{NC} (unavailable - suggestion failed)")
    print(f"  {BLUE}[2]{NC} Enter branch name manually")
    print(f"  {YELLOW}[3]{NC} Commit directly on {current_branch}")
    print(f"  {RED}[4]{NC} Cancel")
    print()

    try:
        choice = input("Choice [1/2/3/4]: ").strip()
    except EOFError:
        choice = '4'

    if choice == '1' and suggested_branch:
        if create_branch(suggested_branch):
            log_success(f"Created and switched to '{suggested_branch}'")
            return suggested_branch
        log_error("Failed to create branch")
        sys.exit(1)

    if choice == '2':
        try:
            manual_branch = input("Enter branch name: ").strip()
        except EOFError:
            manual_branch = ''

        if manual_branch:
            if create_branch(manual_branch):
                log_success(f"Created and switched to '{manual_branch}'")
                return manual_branch
            log_error("Failed to create branch")
            sys.exit(1)

        log_warning("No branch name provided. Cancelled.")
        sys.exit(0)

    if choice == '3':
        log_info(f"Continuing on {current_branch}...")
        return current_branch

    log_warning("Cancelled")
    sys.exit(0)


def handle_pr_flow(current_branch: str, lang: str, push_before_pr: bool) -> None:
    """Generate PR content and create the PR via gh."""
    if is_protected_branch(current_branch):
        log_error("-P requires a non-protected branch. Create or checkout a feature branch first.")
        sys.exit(1)

    base_branch = detect_base_branch()
    if not base_branch:
        log_error("Could not detect base branch. Cannot create PR.")
        sys.exit(1)

    commits_ahead = get_commits_ahead(base_branch)
    if commits_ahead == 0:
        log_warning(f"No commits ahead of {base_branch}")
        sys.exit(0)

    if push_before_pr:
        log_info(f"Pushing '{current_branch}' before PR creation...")
        if not push_branch(current_branch):
            log_error("Failed to push branch before PR creation")
            sys.exit(1)
        log_success("Push successful!")

    if not check_gh_installed():
        log_error("gh CLI is not installed. Install with: sudo apt install gh")
        sys.exit(1)
    if not check_gh_authenticated():
        log_error("gh is not authenticated. Run: gh auth login")
        sys.exit(1)

    commits = get_commits_log(base_branch)
    diff = get_diff_against_base(base_branch)
    files_changed = get_files_changed(base_branch)

    if not diff:
        log_warning(f"No changes detected compared to {base_branch}")
        sys.exit(0)

    log_info("Generating PR title and description...")

    pr_title, pr_body = generate_pr_content(
        commits, diff, files_changed,
        current_branch, base_branch,
        lang,
    )

    if not pr_title:
        log_error("Failed to generate PR description")
        sys.exit(1)

    print()
    print(f"{GREEN}PR Title:{NC}")
    print("-" * 40)
    print(pr_title)
    print("-" * 40)
    print()
    print(f"{GREEN}PR Description:{NC}")
    print("-" * 40)
    print(pr_body)
    print("-" * 40)
    print()

    try:
        pr_url = create_pr(pr_title, pr_body, base_branch)
        print()
        log_success("PR created successfully!")
        log_info(f"URL: {pr_url}")
    except RuntimeError as e:
        log_error(f"Failed to create PR: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Generate a branch name and commit message using the prompt utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  auto-commit                    # Generate message and confirm
  auto-commit -y -Y              # Stage all and skip final confirmation
  auto-commit -s -Y              # Use only staged files
  auto-commit -f                 # Stay on current branch even if protected
  auto-commit -y -Y -p           # Stage, commit, and push automatically
  auto-commit -y -Y -p -P        # Stage, commit, push, and create a PR
  auto-commit -P                 # Create a PR from the current branch when it already has commits
  auto-commit -l pt-br           # Generate message in Portuguese
'''
    )

    parser.add_argument('-f', '--force', action='store_true',
                        help='Stay on the current branch on protected branches')
    parser.add_argument('-y', '-a', '--add', action='store_true',
                        help='Automatically stage all files')
    parser.add_argument('-Y', '--yes', dest='yes_commit', action='store_true',
                        help='Skip final commit confirmation')
    parser.add_argument('-s', '--staged-only', action='store_true',
                        help='Only use files already staged')
    parser.add_argument('-p', '--push', action='store_true',
                        help='Push the current branch after committing')
    parser.add_argument('-P', '--pr', action='store_true',
                        help='Create a PR with gh after pushing (requires -p)')
    parser.add_argument('-l', '--lang', type=str,
                        default=get_language('auto-commit'),
                        help=f'Output language (default: {get_language("auto-commit")})')
    add_llm_request_arguments(parser)

    args = parser.parse_args()

    if args.pr and not args.push:
        log_error("-P requires -p so the branch is pushed before creating the PR")
        sys.exit(1)

    if not is_git_repo():
        log_error("Not inside a git repository")
        sys.exit(1)

    os.chdir(get_repo_root())

    current_branch = get_current_branch()
    on_protected_branch = is_protected_branch(current_branch)

    log_info("Checking for uncommitted changes...")

    staged = get_staged_files()
    unstaged = get_unstaged_files()
    untracked = get_untracked_files()

    if not staged and not unstaged and not untracked:
        if args.pr:
            handle_pr_flow(current_branch, args.lang, args.push)
            return

        log_warning("No changes to commit")
        sys.exit(0)

    print()
    log_info("Changes summary:")

    if staged:
        print(f"{GREEN}Staged:{NC}")
        for f in staged.split('\n'):
            print(f"  {f}")

    if unstaged:
        print(f"{YELLOW}Modified:{NC}")
        for f in unstaged.split('\n'):
            print(f"  {f}")

    if untracked:
        print(f"{RED}Untracked:{NC}")
        for f in untracked.split('\n'):
            print(f"  {f}")

    print()

    if unstaged or untracked:
        if args.staged_only:
            log_info("Staged-only mode (--staged-only); ignoring unstaged and untracked files.")
        elif args.add:
            log_info("Staging all files (--add)...")
            stage_all_files()
        else:
            try:
                reply = input("Stage all files? (y/N) ").strip().lower()
            except EOFError:
                reply = 'n'

            if reply == 'y':
                log_info("Staging files...")
                stage_all_files()
            elif not staged:
                log_warning("No files staged. Aborting.")
                sys.exit(0)

    log_info("Generating diff for analysis...")
    diff = get_staged_diff()

    if not diff:
        log_warning("No staged changes to commit")
        sys.exit(0)

    recent_commits = get_recent_commits(5)
    name_status = get_staged_name_status()
    llm_options = {
        "reasoning_effort": args.reasoning_effort,
        "service_tier": args.service_tier,
    }

    try:
        plan = generate_commit_plan(
            diff,
            name_status,
            recent_commits,
            current_branch,
            args.lang,
            **llm_options,
        )
    except Exception as e:
        log_error(f"Failed to generate commit plan: {e}")
        sys.exit(1)

    branch_name = plan.get('branch_name', '')
    commit_msg = plan.get('commit_message', '')

    auto_pr_from_protected = (
        args.add
        and args.yes_commit
        and args.push
        and args.pr
        and not args.force
    )

    if on_protected_branch:
        if args.force:
            log_info(f"Continuing on {current_branch} (--force)...")
        elif auto_pr_from_protected:
            if not branch_name:
                log_error("Could not suggest branch name")
                sys.exit(1)
            log_info(f"Creating branch '{branch_name}' from protected branch '{current_branch}'...")
            if not create_branch(branch_name):
                log_error("Failed to create branch")
                sys.exit(1)
            current_branch = branch_name
            on_protected_branch = is_protected_branch(current_branch)
            log_success(f"Created and switched to '{current_branch}'")
        else:
            current_branch = handle_protected_branch(current_branch, branch_name)
            on_protected_branch = is_protected_branch(current_branch)

    if args.pr and on_protected_branch:
        log_error("-P requires a non-protected branch. Create or checkout a feature branch first.")
        sys.exit(1)

    if not commit_msg:
        log_error("Failed to generate commit message")
        sys.exit(1)

    print(f"{GREEN}Generated commit message:{NC}")
    print("-" * 40)
    print(commit_msg)
    print("-" * 40)
    print()

    if not args.yes_commit:
        try:
            reply = input("Confirm commit with this message? (Y/n) ").strip().lower()
        except EOFError:
            reply = 'n'

        if reply == 'n':
            log_warning("Commit cancelled")
            sys.exit(0)

    log_info("Committing...")
    create_commit(commit_msg)

    print()
    log_success("Commit successful!")
    log_info(f"Latest commit: {get_latest_commit()}")

    if args.push or args.pr:
        if args.push:
            log_info(f"Pushing '{current_branch}'...")
            if not push_branch(current_branch):
                log_error("Failed to push branch")
                sys.exit(1)
            log_success("Push successful!")

        if args.pr:
            handle_pr_flow(current_branch, args.lang, False)


if __name__ == '__main__':
    main()
