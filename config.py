# config parameters
MODEL_NAME = 'gates'
STYLE_IMG = '/home/ubuntu/dataset/styles/1/style1.jpg'
CONTENT_IMG = './imgs/content/content2.jpg'
STYLE_IMG_PATH = '/home/ubuntu/dataset/styles'
CONTENT_IMG_PATH = '/home/ubuntu/dataset/COCO_2017'

epochs = 4
batch_size = 4
learning_rate = 1e-3
content_img_size = 256
style_img_size = 256

style_loss_weights = [x*1e3 for x in [.35, .35, .15, .15]]
# style_loss_weights = [5, 5, 5, 5]
content_loss_weights = [0, 1, 0, 0]
reg_loss_weight = 1e-7

ckpt_dir='./ckpt/msgnet'
log_dir='./logs/msgnet'
