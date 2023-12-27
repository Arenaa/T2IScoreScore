from abc import ABC, abstractmethod
import click
import torch
from PIL import Image
from utils import csv_line_map, load_pretrained_model, get_model_name_from_path, get_mplug_answer


class AbstractVQAModel(ABC):
    @abstractmethod
    def process_image(self, image_path):
        pass

    @abstractmethod
    def generate(self, model_inputs):
        pass

    @abstractmethod
    def run_vqa(self, text_prompt, image_path):
        pass


def csv_line_map(line):
    return line.strip().split(",")


@click.command()
@click.option("-q", default="HalluVisionFull/HalluVision_TIFA_Q.csv")
@click.option("-o", default="output_csvs/a_mplug_tifa.csv")
@click.option("-b", default="HalluVisionFull/Final-HalluVision/")
@click.option("-s", default="0")
@click.option("-e", default=":")
def get_answers(q, o, b, s, e):
    # Generate the output CSV filename based on the start and end indices
    if s != "0" or e != ":":
        o = f"{o}.{s}-{e}.csv"

    # Load questions from the CSV file
    questions = list(map(csv_line_map, open(q, "r").readlines()))[1:]

    # Set up the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained model
    print("Loading model...")
    model_path = 'MAGAer13/mplug-owl2-llama2-7b'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, load_8bit=False, load_4bit=True, device=device
    )
    print("Model loaded!")

    # Load all images from the 'HalluVisionAll.csv' file
    all_images_list = list(map(csv_line_map, open("HalluVisionFull/HalluVisionAll.csv", "r").readlines()))

    # Adjust the range based on start and end indices
    s, e = max(int(s), 1), int(e) if e != ":" else len(all_images_list)
    all_images_list = all_images_list[s:e]

    # Iterate over all images
    out_lines = []
    fail_imgs = []

    for all_img_line_no in range(len(all_images_list)):
        image_line = all_images_list[all_img_line_no]
        this_id, _, this_fname, *_ = image_line

        # Filter questions for the current image
        question_set = filter(lambda x: int(x[0]) == int(this_id), questions)

        for question_line in question_set:
            _, question_id, question, _ = question_line

            try:
                # Get the answer using the 'get_mplug_answer' function
                answer = get_mplug_answer(question, b + this_fname, image_processor, model, tokenizer)

                # Construct the output line
                out_line = f"{this_id},{this_fname},{question_id},{answer.replace(',', '').strip()}"
                print(out_line)
                out_lines.append(out_line + "\n")
            except FileNotFoundError:
                print(f"File {this_fname} not found")
                fail_imgs.append(f"{this_id},{this_fname}\n")

    # Write the results to the output files
    with open(o, "w") as f:
        f.writelines(out_lines)
    with open(o + ".fail.csv", "w") as f:
        f.writelines(fail_imgs)

if __name__ == "__main__":
    get_answers()
