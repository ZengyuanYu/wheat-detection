import cv2
import os

def video2img(video_path, img_save_path, save_num=1):
    """
    video_path：待分解源视频
    img_save_path: 视频分解图片存储位置
    save_num: 每隔多少帧存储一次
    """
    vc = cv2.VideoCapture(video_path)
    c = 1
    fps = vc.get(cv2.CAP_PROP_FPS)
    print(fps)
    if vc.isOpened():
        rval,frame=vc.read()
    else:
        rval=False
    while rval:
        rval,frame=vc.read()
        if c % save_num == 0:
            cv2.imwrite(img_save_path+str(c)+'.jpg',frame)
        else:
            pass
        # print(c)
        c = c + 1
        cv2.waitKey(1)
    vc.release()


def img2video(img_save_path, video_save_path, fps=30):
    """
    img_save_path: 待转视频图片文件夹
    video_save_path: 视频存储位置
    fps: 帧率 default:30
    """
    fps = 30
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(filename=video_save_path, fourcc=fourcc, fps=fps, frameSize=(1920, 1080))
    for i in range(0,1426):
        p = i
        if os.path.exists(img_save_path+str(p)+'.jpg'):    #判断图片是否存在
            img = cv2.imread(filename=img_save_path+str(p)+'.jpg')
            cv2.waitKey(100)
            video_writer.write(img)
            print(str(p) + '.jpg' + ' done!')
        else:
            pass
    video_writer.release()


if __name__ == '__main__':
    # video2img("./IMG_4337.MOV", './video2img/', 1)
    img2video('./video2img/', './result.avi', 30)