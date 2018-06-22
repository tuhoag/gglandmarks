from gglandmarks.models import MyResNets

def main(data_path='./data/landmarks_recognition', image_original_size=(128, 128), model_dir='./logs/finetune'):    
    MyResNets.finetune(data_path=data_path, image_original_size=image_original_size, model_dir=model_dir)