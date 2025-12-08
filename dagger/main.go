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

	// Connect to Dagger Engine — use Docker Hub mirror for CI reliability
	os.Setenv("_EXPERIMENTAL_DAGGER_ENGINE_IMAGE", "docker.io/dagger/engine:v0.19.8")

	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))

	if err != nil {
		panic(err)
	}
	defer client.Close()

	// Load the project directory (excluding virtualenv/cache)
	project := client.Host().Directory("..", dagger.HostDirectoryOpts{
		Exclude: []string{"venv/", "__pycache__/", "artifacts-out/"},
		Include: []string{".dvc", ".dvc/cache", ".git", "dvc.yaml", "dvc.lock", "mlops_refactor/", "data/"},
	})

	// Base Python container
	python := client.Container().
		From("python:3.11").
		WithMountedDirectory("/work", project).
		WithWorkdir("/work")

	// ---------- INSTALL DEPENDENCIES ----------
	fmt.Println("=== INSTALLING PYTHON DEPENDENCIES ===")
	python = python.WithExec([]string{"pip", "install", "-r", "mlops_refactor/requirements.txt"})

	// ---------- RUN TRAINING PIPELINE ----------
	fmt.Println("=== RUNNING TRAINING PIPELINE ===")
	trained := python.WithExec([]string{"python", "-m", "mlops_refactor.pipeline.train_pipeline"})

	// ---------- CAPTURE OUTPUT ----------
	fmt.Println("=== TRAIN PIPELINE OUTPUT ===")
	output, err := trained.Stdout(ctx)
	if err != nil {
		panic(err)
	}
	fmt.Print(output)
	fmt.Println("=============================")

	// ---------- EXPORT ARTIFACTS ----------
	artifacts := trained.Directory("artifacts")

	fmt.Println("=== EXPORTING ARTIFACTS ===")
	_, err = artifacts.Export(ctx, filepath.Join(".", "artifacts-out"))
	if err != nil {
		panic(err)
	}

	fmt.Println("✅ DONE — Artifacts exported to artifacts-out/")
}
