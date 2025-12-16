package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"dagger.io/dagger"
)

// Main function to run the Dagger pipeline
func main() {
	ctx := context.Background()

	// 1. Connect to the Dagger Engine
	client, err := connectDagger(ctx)
	if err != nil {
		fmt.Printf("Error connecting to Dagger: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	// 2. Set up the Container and Dependencies
	container, err := setupContainer(client, ctx)
	if err != nil {
		fmt.Printf("Error setting up container: %v\n", err)
		os.Exit(1)
	}

	// 3. Run the Training Pipeline
	trainedContainer, err := runTrainingPipeline(container, ctx)
	if err != nil {
		fmt.Printf("Error running training pipeline: %v\n", err)
		os.Exit(1)
	}

	// 4. Capture and Print Output
	if err := captureAndPrintOutput(trainedContainer, ctx); err != nil {
		fmt.Printf("Error capturing output: %v\n", err)
		os.Exit(1)
	}

	// 5. Export Artifacts
	if err := exportArtifacts(trainedContainer, ctx); err != nil {
		fmt.Printf("Error exporting artifacts: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("✅ DONE — Pipeline completed successfully!")
}

// connectDagger establishes a connection to the Dagger Engine.
func connectDagger(ctx context.Context) (*dagger.Client, error) {
	fmt.Println("=== CONNECTING TO DAGGER ENGINE ===")
	// Use Docker Hub mirror for CI reliability
	os.Setenv("_EXPERIMENTAL_DAGGER_ENGINE_IMAGE", "docker.io/dagger/engine:v0.19.8")

	return dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
}

// setupContainer loads the project, creates the base Python container, and installs dependencies.
func setupContainer(client *dagger.Client, ctx context.Context) (*dagger.Container, error) {
	// Load the project directory (excluding virtualenv/cache)
	project := client.Host().Directory("..", dagger.HostDirectoryOpts{
		Exclude: []string{"venv/", "__pycache__/", "artifacts-out/"},
		Include: []string{".dvc", ".dvc/cache", ".git", "dvc.yaml", "dvc.lock", "mlops_refactor/", "data/"},
	})

	// Base Python container setup
	python := client.Container().
		From("python:3.11").
		WithMountedDirectory("/work", project).
		WithWorkdir("/work")

	// Install Dependencies
	fmt.Println("=== INSTALLING PYTHON DEPENDENCIES ===")
	python = python.WithExec([]string{"pip", "install", "-r", "mlops_refactor/requirements.txt"})

	// Force evaluation of dependency installation before proceeding (optional, but good practice)
	if _, err := python.Stdout(ctx); err != nil {
		return nil, fmt.Errorf("failed to install dependencies: %w", err)
	}

	return python, nil
}

// runTrainingPipeline executes the training script inside the container.
func runTrainingPipeline(baseContainer *dagger.Container, ctx context.Context) (*dagger.Container, error) {
	fmt.Println("=== RUNNING TRAINING PIPELINE ===")
	// This creates a new container layer with the training execution
	trained := baseContainer.WithExec([]string{"python", "-m", "mlops_refactor.pipeline.train_pipeline"})
	return trained, nil
}

// captureAndPrintOutput retrieves and prints the standard output from the training run.
func captureAndPrintOutput(trainedContainer *dagger.Container, ctx context.Context) error {
	fmt.Println("=== TRAIN PIPELINE OUTPUT ===")
	output, err := trainedContainer.Stdout(ctx)
	if err != nil {
		return fmt.Errorf("failed to get container stdout: %w", err)
	}
	fmt.Print(output)
	fmt.Println("=============================")
	return nil
}

// exportArtifacts extracts the 'artifacts' directory from the trained container to the host filesystem.
func exportArtifacts(trainedContainer *dagger.Container, ctx context.Context) error {
	artifacts := trainedContainer.Directory("artifacts")

	fmt.Println("=== EXPORTING ARTIFACTS ===")
	_, err := artifacts.Export(ctx, filepath.Join(".", "artifacts-out"))
	if err != nil {
		return fmt.Errorf("failed to export artifacts: %w", err)
	}

	fmt.Println("Artifacts exported to artifacts-out/")
	return nil
}
