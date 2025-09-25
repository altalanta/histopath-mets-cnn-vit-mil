"""Command-line interface for histopathology models."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import torch
from rich.console import Console
from rich.progress import track

from .inference import run_embed_pipeline, load_model_from_checkpoint
from .utils.seed import seed_everything

app = typer.Typer(help="Histopathology ML CLI")
console = Console()


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_status() -> dict:
    """Get git repository status."""
    try:
        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        has_uncommitted = bool(result.stdout.strip())
        
        # Get current branch
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        branch = branch_result.stdout.strip()
        
        return {
            "sha": get_git_sha(),
            "branch": branch,
            "has_uncommitted_changes": has_uncommitted,
            "uncommitted_files": result.stdout.strip().split('\n') if has_uncommitted else []
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "sha": "unknown",
            "branch": "unknown", 
            "has_uncommitted_changes": False,
            "uncommitted_files": []
        }


def compute_schema_hash(df: pd.DataFrame) -> str:
    """Compute schema hash for embedding versioning."""
    schema_info = {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "shape": list(df.shape),
        "index_name": df.index.name,
    }
    
    schema_str = json.dumps(schema_info, sort_keys=True)
    return hashlib.sha256(schema_str.encode()).hexdigest()[:8]


def compute_model_checksum(model_path: Path) -> str:
    """Compute checksum of model file."""
    if not model_path.exists():
        return "file_not_found"
    
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()[:16]


@app.command()
def embed(
    tiles_dir: str = typer.Argument(..., help="Directory containing tile images or path to tar file"),
    model_path: str = typer.Argument(..., help="Path to model checkpoint"),
    output: str = typer.Option("embeddings.parquet", help="Output file path"),
    config_path: Optional[str] = typer.Option(None, help="Path to config file"),
    batch_size: int = typer.Option(32, help="Batch size for inference"),
    device: str = typer.Option("auto", help="Device to use (auto, cpu, cuda)"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    overwrite: bool = typer.Option(False, help="Overwrite existing output files"),
    include_metadata: bool = typer.Option(True, help="Include sample metadata in output"),
) -> None:
    """Generate versioned embeddings with comprehensive manifest."""
    
    # Set up paths
    tiles_path = Path(tiles_dir)
    model_path_obj = Path(model_path)
    config_path_obj = Path(config_path) if config_path else None
    
    # Validate inputs
    if not tiles_path.exists():
        console.print(f"[red]Error: Tiles directory/file not found: {tiles_path}[/red]")
        raise typer.Exit(1)
    
    if not model_path_obj.exists():
        console.print(f"[red]Error: Model checkpoint not found: {model_path_obj}[/red]")
        raise typer.Exit(1)
    
    if config_path_obj and not config_path_obj.exists():
        console.print(f"[red]Error: Config file not found: {config_path_obj}[/red]")
        raise typer.Exit(1)
    
    # Set seed for reproducibility
    seed_everything(seed)
    
    console.print(f"[blue]Starting embedding generation...[/blue]")
    console.print(f"Tiles: {tiles_path}")
    console.print(f"Model: {model_path_obj}")
    console.print(f"Seed: {seed}")
    
    try:
        # Generate embeddings
        embeddings_df = run_embed_pipeline(
            tiles_dir=tiles_path,
            model_path=model_path_obj,
            config_path=config_path_obj,
            batch_size=batch_size,
            device=device,
            include_metadata=include_metadata,
        )
        
        console.print(f"[green]Generated embeddings for {len(embeddings_df)} samples[/green]")
        
        # Compute schema hash and create versioned filename
        schema_hash = compute_schema_hash(embeddings_df)
        output_path = Path(output)
        
        # Create versioned filename
        stem = output_path.stem
        suffix = output_path.suffix
        versioned_name = f"{stem}_{schema_hash}{suffix}"
        versioned_path = output_path.parent / versioned_name
        
        # Check if output already exists
        if versioned_path.exists() and not overwrite:
            console.print(f"[yellow]Output file already exists: {versioned_path}[/yellow]")
            console.print("Use --overwrite to replace existing file")
            raise typer.Exit(1)
        
        # Save embeddings
        versioned_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_df.to_parquet(versioned_path)
        console.print(f"[green]Embeddings saved: {versioned_path}[/green]")
        
        # Create comprehensive manifest
        git_info = get_git_status()
        model_checksum = compute_model_checksum(model_path_obj)
        
        manifest = {
            "embedding_file": str(versioned_path.name),
            "schema_hash": schema_hash,
            "timestamp": datetime.now().isoformat(),
            
            # Model information
            "model": {
                "path": str(model_path_obj),
                "checksum": model_checksum,
                "size_bytes": model_path_obj.stat().st_size if model_path_obj.exists() else None,
            },
            
            # Input information
            "input": {
                "tiles_dir": str(tiles_path),
                "tiles_count": len(embeddings_df),
                "config_path": str(config_path_obj) if config_path_obj else None,
            },
            
            # Generation parameters
            "parameters": {
                "batch_size": batch_size,
                "device": device,
                "seed": seed,
                "include_metadata": include_metadata,
            },
            
            # Environment information
            "environment": {
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "platform": sys.platform,
            },
            
            # Git information
            "git": git_info,
            
            # Data schema
            "schema": {
                "columns": embeddings_df.columns.tolist(),
                "dtypes": embeddings_df.dtypes.astype(str).to_dict(),
                "shape": list(embeddings_df.shape),
                "index_name": embeddings_df.index.name,
            },
            
            # Statistics
            "statistics": {
                "embedding_dim": len([col for col in embeddings_df.columns if col.startswith("emb_")]),
                "mean_embedding_norm": float(embeddings_df.filter(regex="^emb_").pow(2).sum(axis=1).sqrt().mean())
                if any(col.startswith("emb_") for col in embeddings_df.columns) else None,
            }
        }
        
        # Save manifest
        manifest_path = versioned_path.with_suffix(".manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        console.print(f"[green]Manifest saved: {manifest_path}[/green]")
        
        # Display summary
        console.print("\n[bold blue]Generation Summary:[/bold blue]")
        console.print(f"  Samples processed: {len(embeddings_df)}")
        console.print(f"  Embedding dimension: {manifest['statistics']['embedding_dim']}")
        console.print(f"  Schema hash: {schema_hash}")
        console.print(f"  Model checksum: {model_checksum}")
        console.print(f"  Git SHA: {git_info['sha']}")
        
        if git_info["has_uncommitted_changes"]:
            console.print("  [yellow]⚠️  Uncommitted changes detected[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error during embedding generation: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def validate_embeddings(
    embeddings_path: str = typer.Argument(..., help="Path to embeddings parquet file"),
    manifest_path: Optional[str] = typer.Option(None, help="Path to manifest file (auto-detected if None)"),
) -> None:
    """Validate embeddings file against its manifest."""
    
    embeddings_path_obj = Path(embeddings_path)
    
    if not embeddings_path_obj.exists():
        console.print(f"[red]Error: Embeddings file not found: {embeddings_path_obj}[/red]")
        raise typer.Exit(1)
    
    # Auto-detect manifest path if not provided
    if manifest_path is None:
        manifest_path_obj = embeddings_path_obj.with_suffix(".manifest.json")
    else:
        manifest_path_obj = Path(manifest_path)
    
    if not manifest_path_obj.exists():
        console.print(f"[red]Error: Manifest file not found: {manifest_path_obj}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Validating embeddings...[/blue]")
    
    try:
        # Load embeddings and manifest
        embeddings_df = pd.read_parquet(embeddings_path_obj)
        
        with open(manifest_path_obj) as f:
            manifest = json.load(f)
        
        # Validate schema hash
        current_schema_hash = compute_schema_hash(embeddings_df)
        expected_schema_hash = manifest.get("schema_hash", "unknown")
        
        if current_schema_hash != expected_schema_hash:
            console.print(f"[red]❌ Schema hash mismatch![/red]")
            console.print(f"  Expected: {expected_schema_hash}")
            console.print(f"  Current:  {current_schema_hash}")
            raise typer.Exit(1)
        
        # Validate shape
        expected_shape = manifest.get("schema", {}).get("shape", [])
        if list(embeddings_df.shape) != expected_shape:
            console.print(f"[red]❌ Shape mismatch![/red]")
            console.print(f"  Expected: {expected_shape}")
            console.print(f"  Current:  {list(embeddings_df.shape)}")
            raise typer.Exit(1)
        
        # Validate columns
        expected_columns = manifest.get("schema", {}).get("columns", [])
        if embeddings_df.columns.tolist() != expected_columns:
            console.print(f"[red]❌ Column mismatch![/red]")
            console.print(f"  Expected: {expected_columns[:5]}...")
            console.print(f"  Current:  {embeddings_df.columns.tolist()[:5]}...")
            raise typer.Exit(1)
        
        console.print("[green]✅ Embeddings validation passed![/green]")
        console.print(f"  Shape: {embeddings_df.shape}")
        console.print(f"  Schema hash: {current_schema_hash}")
        console.print(f"  Generated: {manifest.get('timestamp', 'unknown')}")
        
    except Exception as e:
        console.print(f"[red]Error during validation: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def info(
    embeddings_path: str = typer.Argument(..., help="Path to embeddings parquet file"),
) -> None:
    """Display information about embeddings file."""
    
    embeddings_path_obj = Path(embeddings_path)
    manifest_path_obj = embeddings_path_obj.with_suffix(".manifest.json")
    
    if not embeddings_path_obj.exists():
        console.print(f"[red]Error: Embeddings file not found: {embeddings_path_obj}[/red]")
        raise typer.Exit(1)
    
    try:
        # Load embeddings
        embeddings_df = pd.read_parquet(embeddings_path_obj)
        
        console.print(f"[bold blue]Embeddings File: {embeddings_path_obj.name}[/bold blue]")
        console.print(f"  Shape: {embeddings_df.shape}")
        console.print(f"  Size: {embeddings_path_obj.stat().st_size / 1024 / 1024:.1f} MB")
        console.print(f"  Columns: {len(embeddings_df.columns)}")
        
        # Show embedding columns
        emb_cols = [col for col in embeddings_df.columns if col.startswith("emb_")]
        if emb_cols:
            console.print(f"  Embedding dimensions: {len(emb_cols)}")
        
        # Load and display manifest if available
        if manifest_path_obj.exists():
            with open(manifest_path_obj) as f:
                manifest = json.load(f)
            
            console.print(f"\n[bold blue]Manifest Information:[/bold blue]")
            console.print(f"  Generated: {manifest.get('timestamp', 'unknown')}")
            console.print(f"  Schema hash: {manifest.get('schema_hash', 'unknown')}")
            console.print(f"  Git SHA: {manifest.get('git', {}).get('sha', 'unknown')}")
            console.print(f"  Model checksum: {manifest.get('model', {}).get('checksum', 'unknown')}")
            
            if manifest.get('git', {}).get('has_uncommitted_changes'):
                console.print("  [yellow]⚠️  Generated with uncommitted changes[/yellow]")
        else:
            console.print(f"\n[yellow]No manifest file found: {manifest_path_obj}[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error reading embeddings: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()