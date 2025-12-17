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

	client, err := connectDagger(ctx)
	if err != nil {
		fmt.Printf("Error connecting to Dagger: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	container, err := setupContainer(client, ctx)
	if err != nil {
		fmt.Printf("Error setting up container: %v\n", err)
		os.Exit(1)
	}

	trainedContainer, err := runTrainingPipeline(container, ctx)
	if err != nil {
		fmt.Printf("Error running training pipeline: %v\n", err)
		os.Exit(1)
	}

	if err := captureAndPrintOutput(trainedContainer, ctx); err != nil {
		fmt.Printf("Error capturing output: %v\n", err)
		os.Exit(1)
	}

	if err := exportArtifacts(trainedContainer, ctx); err != nil {
		fmt.Printf("Error exporting artifacts: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("✅ DONE — Pipeline completed successfully!")
}

func connectDagger(ctx context.Context) (*dagger.Client, error) {
	fmt.Println("=== CONNECTING TO DAGGER ENGINE ===")
	os.Setenv("_EXPERIMENTAL_DAGGER_ENGINE_IMAGE", "docker.io/dagger/engine:v0.19.8")

	return dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
}

func setupContainer(client *dagger.Client, ctx context.Context) (*dagger.Container, error) {
	project := client.Host().Directory("..", dagger.HostDirectoryOpts{
		Exclude: []string{"venv/", "__pycache__/", "artifacts-out/"},
		Include: []string{".dvc", ".dvc/cache", ".git", "dvc.yaml", "dvc.lock", "mlops_refactor/", "data/"},
	})

	python := client.Container().
		From("python:3.11").
		WithMountedDirectory("/work", project).
		WithWorkdir("/work")

	fmt.Println("=== INSTALLING PYTHON DEPENDENCIES ===")
	python = python.WithExec([]string{"pip", "install", "-r", "mlops_refactor/requirements.txt"})

	return python, nil
}

func runTrainingPipeline(baseContainer *dagger.Container, ctx context.Context) (*dagger.Container, error) {
	fmt.Println("=== RUNNING TRAINING PIPELINE ===")
	trained := baseContainer.WithExec([]string{"python", "-m", "mlops_refactor.src.pipeline.train_pipeline"})
	return trained, nil
}

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
