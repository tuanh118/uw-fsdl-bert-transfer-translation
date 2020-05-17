from transformers import pipeline

class PipelineModel():

    def __init__(self, original_language, goal_language):
        save_directory = f"translation_{original_language}_to_{goal_language}"

        self.translator = pipeline(save_directory)

    def forward(self, input):
        output = self.translator(input, max_length=400)

        return [ f"{l['translation_text']}\n" for l in output ]
