import os
import argparse
import numpy as np
from BudaOCR.Modules import CTCDataset, EasterNetwork, CRNNNetwork, WylieEncoder, StackEncoder, ModelTester
from BudaOCR.Utils import build_distribution_paths, read_distribution, read_ocr_model_config


os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()
    input_dir = args.input
    chkpt_dir = args.checkpoint

    config_file = f"{chkpt_dir}/model_config.json"
    distribution_file = f"{chkpt_dir}/data.distribution"

    assert os.path.isfile(config_file)
    assert os.path.isfile(distribution_file)

    checkpoint, architecture, encoder, input_width, input_height, charset = read_ocr_model_config(config_file)
    train_samples, valid_samples, test_samples = read_distribution(distribution_file)
    test_images, test_label_paths = build_distribution_paths(input_dir, test_samples) 

    if encoder == "wylie":
        label_encoder = WylieEncoder(charset)
    else:
        label_encoder = StackEncoder(charset)

    test_labels = [label_encoder.read_label(x) for x in test_label_paths]
    num_classes = label_encoder.num_classes()

    if architecture == "Easter2":
        network = EasterNetwork(input_width, input_height, num_classes)
    else:
        network = CRNNNetwork(input_width, input_height, num_classes)

    test_dataset = CTCDataset(
            images=test_images,
            labels=test_labels,
            label_encoder=label_encoder,
            img_height=input_height,
            img_width=input_width
        )

    model_tester = ModelTester(network, label_encoder)
    cer_scores = model_tester.evaluate(test_dataset, test_label_paths)
    cer_values = list(cer_scores.values())

    print(f"Mean CER: {np.mean(cer_values)}")
    print(f"Max CER: {np.max(cer_values)}")
    print(f"Min CER: {np.min(cer_values)}")

    score_file = os.path.join(chkpt_dir, "cer_scores.txt")

    with open(score_file, "w", encoding="utf-8") as f:
        for sample, value in cer_scores.items():
            f.write(f"{sample} - {value}/n")