package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	// Connect to Dagger Engine
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
	if err != nil {
		panic(err)
	}
	defer client.Close()

	// Load the host directory
	project := client.Host().Directory("..", dagger.HostDirectoryOpts{
	Exclude: []string{"venv/", "__pycache__/", "artifacts-out/"},
	Include: []string{".dvc", ".dvc/cache", ".git", "dvc.yaml", "dvc.lock", "mlops_refactor/", "data/"},
})

	// Define the base Python container
	python := client.Container().
		From("python:3.11").
		WithMountedDirectory("/work", project).
		WithWorkdir("/work")

	// ---------- DVC PULL DATA ----------
	fmt.Println("=== RUNNING DVC PULL ===")
	python_dvc := python.WithWorkdir("/work")
	python_dvc = python_dvc.WithExec([]string{"pip", "install", "dvc"})
	python_dvc = python_dvc.WithExec([]string{"dvc", "pull", "mlops_refactor/data/raw/raw.csv.dvc"})

	// ---------- PIP INSTALL  ----------
	fmt.Println("=== RUNNING PIP INSTALL ===")
	python = python_dvc.WithExec([]string{"pip", "install", "-r", "mlops_refactor/requirements.txt"})

	// ---------- RUN TRAINING ----------
	fmt.Println("=== RUN TRAINING ===")
	trained := python.WithExec([]string{"python", "-m", "mlops_refactor.pipeline.train_pipeline"})

	// Force Execution
	fmt.Println("=== TRAIN PIPELINE STDOUT ===")
	output, err := trained.Stdout(ctx)
	if err != nil {
		panic(err)
	}
	fmt.Print(output)
	fmt.Println("=============================")

	artifacts := trained.Directory("artifacts")

	fmt.Println("=== EXPORTING ARTIFACTS ===")
	_, err = artifacts.Export(ctx, filepath.Join(".", "artifacts-out"))
	if err != nil {
		panic(err)
	}

	fmt.Println("DONE — Artifacts exported to artifacts-out/")
}
