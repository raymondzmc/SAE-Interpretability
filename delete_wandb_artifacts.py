#!/usr/bin/env python3
"""
Script to delete all artifacts from a specified Weights & Biases project.

Usage:
    python scripts/delete_wandb_artifacts.py --project raymondl/tinystories-1m-hardconcrete [--dry-run]
"""

import argparse
import wandb
from typing import List
import time
from settings import settings


def delete_all_artifacts(project: str, dry_run: bool = False) -> None:
    """Delete all artifacts from a W&B project.
    
    Args:
        project: The W&B project in format "entity/project_name"
        dry_run: If True, only list artifacts without deleting them
    """
    print(f"{'[DRY RUN] ' if dry_run else ''}Deleting all artifacts from project: {project}")
    
    # Login to wandb
    if settings.wandb_api_key:
        wandb.login(key=settings.wandb_api_key)
    else:
        wandb.login()
    
    api = wandb.Api()
    
    try:
        # Get all artifact types in the project
        artifact_types = api.artifact_types(project)
        
        # System-managed artifact types that cannot be deleted
        SYSTEM_MANAGED_TYPES = {
            'wandb-history',  # Run history artifacts - automatically created by W&B
            'wandb-summary',  # Run summary artifacts - automatically created by W&B
        }
        
        total_artifacts = 0
        deleted_artifacts = 0
        skipped_system_artifacts = 0
        
        for artifact_type in artifact_types:
            print(f"\n--- Processing artifact type: {artifact_type.name} ---")
            
            # Skip system-managed artifact types that cannot be deleted
            if artifact_type.name in SYSTEM_MANAGED_TYPES:
                collections = api.artifact_collections(project_name=project, type_name=artifact_type.name)
                system_count = sum(len(list(collection.artifacts())) for collection in collections)
                skipped_system_artifacts += system_count
                total_artifacts += system_count  # Count them in total but mark as skipped
                print(f"  SKIPPED: {system_count} system-managed artifacts (cannot be deleted via API)")
                continue
            
            # Get all artifacts of this type using the correct API
            try:
                # For artifacts, we need to specify the collection name pattern
                # Try to get artifacts by searching for collections in this project
                collections = api.artifact_collections(project_name=project, type_name=artifact_type.name)
                
                type_count = 0
                for collection in collections:
                    print(f"  Processing collection: {collection.name}")
                    artifacts = collection.artifacts()
                    artifact_list = list(artifacts)
                    collection_count = len(artifact_list)
                    type_count += collection_count
                    
                    print(f"  Found {collection_count} artifacts in collection '{collection.name}'")
                    
                    for i, artifact in enumerate(artifact_list, 1):
                        artifact_name = f"{artifact.name}:{artifact.version}"
                        
                        if dry_run:
                            alias_info = f" (aliases: {artifact.aliases})" if artifact.aliases else ""
                            print(f"    [{i}/{collection_count}] Would delete: {artifact_name}{alias_info}")
                        else:
                            try:
                                print(f"    [{i}/{collection_count}] Deleting: {artifact_name}")
                                
                                # Delete the artifact with aliases using the delete_aliases parameter
                                if artifact.aliases:
                                    print(f"      Removing aliases: {artifact.aliases}")
                                    artifact.delete(delete_aliases=True)
                                else:
                                    artifact.delete()
                                
                                deleted_artifacts += 1
                                print(f"      Successfully deleted: {artifact_name}")
                                
                                # Add a small delay to avoid rate limiting
                                time.sleep(0.1)
                                
                            except Exception as e:
                                print(f"      ERROR deleting {artifact_name}: {e}")
                
                total_artifacts += type_count
                print(f"Total artifacts found for type '{artifact_type.name}': {type_count}")
                
            except Exception as e:
                print(f"  ERROR accessing artifacts of type '{artifact_type.name}': {e}")
                continue
        
        print(f"\n=== SUMMARY ===")
        print(f"Total artifacts found: {total_artifacts}")
        print(f"System-managed artifacts (skipped): {skipped_system_artifacts}")
        print(f"User artifacts: {total_artifacts - skipped_system_artifacts}")
        
        if dry_run:
            print(f"User artifacts that would be deleted: {total_artifacts - skipped_system_artifacts}")
            print("\nRe-run without --dry-run to actually delete the user artifacts.")
            print("Note: System-managed artifacts (like wandb-history) cannot be deleted via API.")
        else:
            print(f"Successfully deleted: {deleted_artifacts}")
            print(f"Failed to delete: {total_artifacts - skipped_system_artifacts - deleted_artifacts}")
            if skipped_system_artifacts > 0:
                print(f"System-managed artifacts (cannot be deleted): {skipped_system_artifacts}")
            
    except Exception as e:
        print(f"ERROR: Failed to access project '{project}': {e}")
        print("Make sure the project exists and you have the necessary permissions.")


def main():
    parser = argparse.ArgumentParser(
        description="Delete all artifacts from a Weights & Biases project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run to see what would be deleted
    python scripts/delete_wandb_artifacts.py --project raymondl/tinystories-1m-hardconcrete --dry-run
    
    # Actually delete all artifacts
    python scripts/delete_wandb_artifacts.py --project raymondl/tinystories-1m-hardconcrete
        """
    )
    
    parser.add_argument(
        "--project", 
        type=str, 
        required=True,
        help="W&B project in format 'entity/project_name'"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list artifacts without deleting them"
    )
    
    args = parser.parse_args()
    
    # Validate project format
    if "/" not in args.project:
        print("ERROR: Project must be in format 'entity/project_name'")
        return 1
    
    # Confirm deletion if not dry run
    if not args.dry_run:
        print(f"WARNING: This will permanently delete ALL artifacts from project '{args.project}'")
        response = input("Are you sure you want to continue? (yes/no): ").lower().strip()
        
        if response not in ["yes", "y"]:
            print("Operation cancelled.")
            return 0
    
    delete_all_artifacts(args.project, args.dry_run)
    return 0


if __name__ == "__main__":
    exit(main()) 