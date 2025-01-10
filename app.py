import gradio as gr
import os
import torch

from model import create_effnetb2_model

effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=5)

effnetb2.load_state_dict(torch.load(
    f="treasure_hunt_effnetb2.pth",
    map_location=torch.device("cpu")
))

class_names = ['clock', 'fork', 'mug', 'remote_control', 'toilet_paper']
hint_list = ["In a realm where shadows and light intertwine, I dwell. I am the keeper of moments, yet I am bound to the eternal loop. My guardians are silent but vigilant, marking the journey from dawn to dusk. Unseen yet ever-present, I reveal the unspoken rhythm of time. What am I?", 
             "I am not a weapon, yet I have many prongs. I am not a musician, but I am often used in ensembles. My purpose is humble, yet essential for a daily ritual. What am I?", 
             "I hold a brew that warms your spirit, yet I am not a cauldron. I am often filled and emptied in quiet rituals of solace. My handle is a small grasp, yet I am always within reach. What am I?", 
             "I am a bridge between the seen and the unseen, a mediator of moving images from afar. My purpose is to command, though I am seldom touched. Through me, choices are made without a direct voice. What am I?", 
             " I am used in private moments of necessity, yet I am rarely spoken of. I am both a barrier and a comfort, designed for a task that is often considered unmentionable. My presence is mundane, but my absence is keenly felt. What am I?"]


def predict(img):
  img = effnetb2_transforms(img).unsqueeze(0)
  effnetb2.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(effnetb2(img), dim=1)
  pred_label = torch.argmax(pred_probs, dim=1)
  label_name = class_names[pred_label]
  return label_name

def check_image_answer(state_index, img):
    hint_index = state_index
    predicted_label = predict(img)
    expected_label = class_names[hint_index]
    
    if predicted_label == expected_label:
        if hint_index < len(hint_list) - 1:
          new_index = hint_index + 1
          return f"Round {new_index + 1}: {hint_list[new_index]}", new_index, None
        else:
          return "Congratulations! You've completed the quiz 🤯", -1, None
    else:
        return (f"Incorrect ☠️! Try again.\n"
                f"Round {hint_index + 1}: {hint_list[hint_index]}", 
                hint_index, None)

def js_to_prefere_the_back_camera_of_mobilephones():
    custom_html = """
    <script>
    const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    
    navigator.mediaDevices.getUserMedia = (constraints) => {
      if (!constraints.video.facingMode) {
        constraints.video.facingMode = {ideal: "environment"};
      }
      return originalGetUserMedia(constraints);
    };
    </script>
    """
    return custom_html
    
with gr.Blocks(head=js_to_prefere_the_back_camera_of_mobilephones()) as demo:
  gr.Markdown("""
    # Welcome to the Image Recognition Puzzle!📷🤳
    
    In this puzzle, you will be given a series of hints. Your task is to upload an image that matches the hint provided for each round. 
    If you correctly identify the image, you will advance to the next round. Complete all rounds to finish the quiz.
    
    Good luck!
    """)

  hint_index = gr.State(value=0)
  is_active = gr.State(value=True)

  hint_textbox = gr.Textbox(label="Hint🤔", value=f"Round 1: {hint_list[0]}", interactive=False)
  image_input = gr.Image(type="pil", label="📸")
  
  submit_button = gr.Button("Submit Image 📲")
  submit_button.click(check_image_answer, 
                        inputs=[hint_index, image_input], 
                        outputs=[hint_textbox, hint_index, image_input])
  

demo.launch(debug=False,
           share=True)
