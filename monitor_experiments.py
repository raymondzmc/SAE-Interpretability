#!/usr/bin/env python3
"""
Monitor active SAE experiments running in tmux sessions.

Usage:
    python monitor_experiments.py           # Show all active experiments
    python monitor_experiments.py --kill    # Kill all experiment sessions
    python monitor_experiments.py --attach  # Interactively attach to a session
"""

import subprocess
import argparse
import sys
import time
from typing import List, Dict, Optional


def get_tmux_sessions() -> List[Dict[str, str]]:
    """Get list of all tmux sessions."""
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}:#{session_created}:#{session_attached}"],
            capture_output=True, text=True, check=True
        )
        
        sessions = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(':')
                if len(parts) >= 3:
                    sessions.append({
                        'name': parts[0],
                        'created': parts[1],
                        'attached': parts[2] == '1'
                    })
        return sessions
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def get_experiment_sessions() -> List[Dict[str, str]]:
    """Get only experiment-related tmux sessions."""
    sessions = get_tmux_sessions()
    return [s for s in sessions if s['name'].startswith('exp_')]


def show_session_status(session_name: str) -> Optional[str]:
    """Get the last line of output from a tmux session."""
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session_name, "-p"],
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split('\n')
            return lines[-1] if lines else None
    except Exception:
        pass
    return None


def monitor_experiments():
    """Show status of all running experiments."""
    experiments = get_experiment_sessions()
    
    if not experiments:
        print("No active experiment sessions found.")
        print("Run experiments with: python run_experiments.py ...")
        return
    
    print(f"Active Experiment Sessions ({len(experiments)}):")
    print("=" * 80)
    
    for exp in experiments:
        # Parse experiment info from session name
        name_parts = exp['name'].replace('exp_', '').split('_')
        experiment_name = '_'.join(name_parts[:-1]) if len(name_parts) > 1 else name_parts[0]
        device = name_parts[-1] if len(name_parts) > 1 else 'unknown'
        
        status_indicator = "🟢" if exp['attached'] else "⚪"
        
        print(f"{status_indicator} {exp['name']}")
        print(f"   Experiment: {experiment_name}")
        print(f"   Device: {device}")
        print(f"   Attached: {'Yes' if exp['attached'] else 'No'}")
        
        # Try to get last output line
        last_output = show_session_status(exp['name'])
        if last_output:
            # Truncate long lines
            if len(last_output) > 100:
                last_output = last_output[:97] + "..."
            print(f"   Last output: {last_output}")
        
        print(f"   Monitor: tmux attach -t {exp['name']}")
        print()


def kill_experiment_sessions():
    """Kill all experiment sessions."""
    experiments = get_experiment_sessions()
    
    if not experiments:
        print("No active experiment sessions to kill.")
        return
    
    print(f"Killing {len(experiments)} experiment sessions...")
    
    for exp in experiments:
        try:
            subprocess.run(["tmux", "kill-session", "-t", exp['name']], 
                         capture_output=True, check=True)
            print(f"✅ Killed: {exp['name']}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to kill: {exp['name']}")


def interactive_attach():
    """Interactively choose and attach to an experiment session."""
    experiments = get_experiment_sessions()
    
    if not experiments:
        print("No active experiment sessions found.")
        return
    
    print("Active experiment sessions:")
    for i, exp in enumerate(experiments):
        name_parts = exp['name'].replace('exp_', '').split('_')
        experiment_name = '_'.join(name_parts[:-1]) if len(name_parts) > 1 else name_parts[0]
        device = name_parts[-1] if len(name_parts) > 1 else 'unknown'
        
        print(f"  {i+1}. {experiment_name} on {device} ({'attached' if exp['attached'] else 'detached'})")
    
    try:
        choice = input(f"\nSelect session to attach (1-{len(experiments)}, or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            return
        
        session_idx = int(choice) - 1
        if 0 <= session_idx < len(experiments):
            session_name = experiments[session_idx]['name']
            print(f"Attaching to {session_name}...")
            print("(Use Ctrl+B, D to detach)")
            
            # Attach to the session
            subprocess.run(["tmux", "attach", "-t", session_name])
        else:
            print("Invalid selection.")
            
    except (ValueError, KeyboardInterrupt):
        print("\nCancelled.")


def watch_experiments():
    """Continuously monitor experiments."""
    print("Watching experiments (Ctrl+C to stop)...")
    print()
    
    try:
        while True:
            # Clear screen
            subprocess.run(["clear"], check=False)
            
            print("SAE Experiment Monitor - Real-time View")
            print("=" * 50)
            print(f"Updated: {time.strftime('%H:%M:%S')}")
            print()
            
            experiments = get_experiment_sessions()
            
            if not experiments:
                print("No active experiments.")
            else:
                for exp in experiments:
                    name_parts = exp['name'].replace('exp_', '').split('_')
                    experiment_name = '_'.join(name_parts[:-1]) if len(name_parts) > 1 else name_parts[0]
                    device = name_parts[-1] if len(name_parts) > 1 else 'unknown'
                    
                    status = "🟢 Running" if not exp['attached'] else "👁️  Monitored"
                    
                    print(f"{status} {experiment_name} ({device})")
                    
                    # Show last output
                    last_output = show_session_status(exp['name'])
                    if last_output:
                        if len(last_output) > 80:
                            last_output = last_output[:77] + "..."
                        print(f"    {last_output}")
                    print()
            
            print("Commands:")
            print("  tmux list-sessions          # List all sessions")
            print("  tmux attach -t <session>    # Attach to specific session")
            print("  python monitor_experiments.py --kill  # Kill all experiments")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nStopping monitor.")


def main():
    parser = argparse.ArgumentParser(description="Monitor SAE experiments running in tmux")
    parser.add_argument("--kill", action="store_true", help="Kill all experiment sessions")
    parser.add_argument("--attach", action="store_true", help="Interactively attach to a session")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor experiments")
    
    args = parser.parse_args()
    
    # Check if tmux is available
    try:
        subprocess.run(["tmux", "-V"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ tmux is not available. Please install tmux first.")
        sys.exit(1)
    
    if args.kill:
        kill_experiment_sessions()
    elif args.attach:
        interactive_attach()
    elif args.watch:
        watch_experiments()
    else:
        monitor_experiments()


if __name__ == "__main__":
    main() 