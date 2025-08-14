#!/bin/bash
# Script to check disk usage in home directory and identify what can be cleaned

echo "============================================================"
echo "HOME DIRECTORY DISK USAGE ANALYSIS"
echo "============================================================"
echo

# Check overall disk usage
echo "Overall disk usage for $HOME:"
df -h $HOME
echo

# Check top-level directories
echo "Top 10 largest directories in home:"
echo "------------------------------------------------------------"
du -sh ~/.[!.]* ~/* 2>/dev/null | sort -hr | head -20
echo

# Check specific cache directories that are usually safe to clean
echo "============================================================"
echo "CACHE DIRECTORIES (usually safe to clean):"
echo "============================================================"

# Check .config subdirectories
if [ -d ~/.config ]; then
    echo "~/.config/ subdirectories:"
    du -sh ~/.config/* 2>/dev/null | sort -hr | head -10
    echo
fi

# Check .local subdirectories
if [ -d ~/.local ]; then
    echo "~/.local/ subdirectories:"
    du -sh ~/.local/* 2>/dev/null | sort -hr | head -10
    echo
fi

# Check .cache if it exists
if [ -d ~/.cache ]; then
    echo "~/.cache/ contents:"
    du -sh ~/.cache/* 2>/dev/null | sort -hr | head -10
    echo
fi

# Check conda cache
if [ -d ~/.conda ]; then
    echo "Conda cache:"
    du -sh ~/.conda/pkgs 2>/dev/null
    echo "  Can clean with: conda clean --all"
    echo
fi

# Check pip cache
echo "Pip cache locations:"
du -sh ~/.cache/pip 2>/dev/null
du -sh ~/.local/pipx 2>/dev/null
echo "  Can clean with: pip cache purge"
echo

# Check VSCode Server (can be large)
if [ -d ~/.vscode-server ]; then
    echo "VSCode Server:"
    du -sh ~/.vscode-server 2>/dev/null
    du -sh ~/.vscode-server/extensions 2>/dev/null
    du -sh ~/.vscode-server/data 2>/dev/null
    echo "  Note: Only clean if not actively using VSCode Remote"
    echo
fi

# Check for wandb directories
echo "Wandb directories:"
du -sh ~/.config/wandb 2>/dev/null
du -sh ~/.cache/wandb 2>/dev/null
du -sh ~/wandb 2>/dev/null
echo

# Check for tmp files
echo "Temporary files in home:"
find ~ -maxdepth 1 -name "tmp*" -o -name "*.tmp" -o -name "*.swp" 2>/dev/null | while read f; do
    du -sh "$f" 2>/dev/null
done
echo

# Check for large log files
echo "Large log files (>100MB):"
find ~ -name "*.log" -size +100M 2>/dev/null | while read f; do
    du -sh "$f" 2>/dev/null
done
echo

echo "============================================================"
echo "RECOMMENDATIONS:"
echo "============================================================"
echo "1. Clean conda cache: conda clean --all"
echo "2. Clean pip cache: pip cache purge"
echo "3. Remove tmp files: rm -rf ~/tmp* ~/*.tmp ~/*.swp"
echo "4. Clean wandb cache: rm -rf ~/.config/wandb/tmp* ~/.cache/wandb"
echo "5. If not using VSCode: rm -rf ~/.vscode-server"
echo "6. Clean old logs: find ~ -name '*.log' -mtime +7 -delete"
echo
echo "Always review before deleting!" 