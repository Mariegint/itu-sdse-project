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
	project := client.Host().Directory(".", dagger.HostDirectoryOpts{
		Exclude: []string{"venv/", "__pycache__/", "artifacts-out/"}, // Added artifacts-out/ to avoid copying prior output
	})

	// Define the base Python container
	python := client.Container().
		From("python:3.11").
		WithMountedDirectory("/work", project).
		WithWorkdir("/work")

	// ---------- PIP INSTALL  ----------
	fmt.Println("=== RUNNING PIP INSTALL ===")
	// Creates a NEW container object reflecting the changes.
	python = python.WithExec([]string{"pip", "install", "-r", "requirements.txt"})

	// ---------- RUN TRAINING ----------
	fmt.Println("=== RUN TRAINING ===")
	// This container holds the state after training is complete (and 'artifacts' are created)
	trained := python.WithExec([]string{"python", "-m", "mlops_refactor.pipeline.train_pipeline"})

	// Force Execution
	fmt.Println("=== TRAIN PIPELINE STDOUT ===")
	output, err := trained.Stdout(ctx)
	if err != nil {
		panic(err)
	}
	fmt.Print(output) // Print the output from the training run
	fmt.Println("=============================")

	// Get the 'artifacts' directory from the now-executed container
	// 'trained' is used here, ensuring we get the directory created by the training step.
	artifacts := trained.Directory("artifacts")

	// Export the artifacts
	fmt.Println("=== EXPORTING ARTIFACTS ===")
	_, err = artifacts.Export(ctx, filepath.Join(".", "artifacts-out"))
	if err != nil {
		panic(err)
	}

	fmt.Println("DONE — Artifacts exported to artifacts-out/")
}
