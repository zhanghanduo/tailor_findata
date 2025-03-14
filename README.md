# Fine-Tuning phi-4 for Financial Question Answering

This repository is dedicated to enhancing the performance of the [phi-4 model](https://huggingface.co/microsoft/phi-4) through supervised fine-tuning. The goal is to improve the model's ability to retrieve and generate answers for questions in the ConvFinQA test set, a dataset focused on conversational financial question answering.

## Strategy Overview

The chosen strategy involves fine-tuning the phi-4 model using the train set of the ConvFinQA dataset. This approach is selected for its potential to deliver the most accurate results, particularly in generating programs for numerical reasoning tasks. Fine-tuning allows the model to learn specific patterns and nuances of the dataset, ensuring precision and reliability in its outputs.

## Why Fine-Tuning?

- **Accuracy**: Fine-tuning on the train set enables the model to learn the exact patterns required for generating correct answers, especially for tasks involving numerical reasoning.
- **Customization**: Tailors the model to the specific requirements of the ConvFinQA dataset, improving its performance in this niche domain.
- **Reliability**: While resource-intensive, this method is expected to yield the best results in terms of accuracy and consistency.

## Getting Started

To get started with fine-tuning the phi-4 model, follow the instructions provided in the repository. Ensure you have access to the ConvFinQA dataset and the necessary computational resources for training.

For more details on the strategy and analysis, refer to the `docs/strategy_selection.md` file.
