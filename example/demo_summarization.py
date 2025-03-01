from explainer.attribute import Preprocessor, ExplainerHandler
from explainer.explainer import Explainer
from explainer.extra import ModelExplanationTask
import os 

current_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    model_name = "mrm8488/t5-base-finetuned-summarize-news"
    example = [
        {
            "text": "Media playback is unsupported on your device\n18 December 2014 Last updated at\
                    10:28 GMT Malaysia has successfully tackled poverty over the last four decades \
                    by drawing on its rich natural resources.\nAccording to the World Bank, some \
                    49% of Malaysians in 1970 were extremely poor, and that figure has been reduced \
                    to 1% today. However, the government's next challenge is to help the lower income\
                    group to move up to the middle class, the bank says.\nUlrich Zahau, the World Bank's\
                    Southeast Asia director, spoke to the BBC's Jennifer Pak.",
            "summary": "In Malaysia the 'aspirational' low-income part of the population is helping to drive economic \
                        growth through consumption, according to the World Bank.",
        }
    ]
    preprocess_result = Preprocessor(
        input_text=example,
        task_type="summarizations",
        model_name_or_path=model_name,
        tokenizer=model_name,
        model_config_path=os.path.join(current_dir, "../tests/yaml/summarization.yaml"),
    )
    model = preprocess_result.model
    tokenizer = preprocess_result.tokenizer
    inputs_all, inputs = preprocess_result()
    generate_input = {
        "input_ids": inputs["input_ids"].clone(),
        "do_sample": False,
        "top_k": 1,
        "top_p": 0.9,
        "temperature": 0.3,
        "repetition_penalty": 1.3,
        "return_dict_in_generate": True,
        "output_logits": True,
        "eos_token_id": tokenizer.eos_token_id,
        "num_beams": 1,
        "output_hidden_states": False,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_logits": True,
        "max_length": (inputs_all["labels"].shape[-1]) + 1,
    }


    with Explainer(model=model, method="Gradient") as explain:
        model.eval()
        output = model.generate(**generate_input)
        param = {
            "tokenizer": tokenizer,
            "output": output,
            "labels": inputs_all["labels"][0],
            "task_type": ModelExplanationTask.SUMMARIZATIONS,
            "input_ids": inputs_all["input_ids"],
        }

        result = explain.compute(**param)

    not isinstance(result["Prompt"], type(None))
    not isinstance(result["Prediction"], type(None))
    not isinstance(result["Gradient"]["saliency_map"], type(None))
