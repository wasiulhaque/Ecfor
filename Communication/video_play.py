import cv2
# import video_mapper


class VideoPlayerPath:
    def video_process(self, video_path_list):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('merged_drum.avi', fourcc, 25.0, (1920, 1080))
        # print(video_path_list)
        for path in video_path_list:
            cap = cv2.VideoCapture(path)
            # frameToStart = 100
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))
                cv2.imshow('frame', frame)
                ch = 0xFF & cv2.waitKey(15)
                out.write(frame)
                # if ch == 27:
                #     break
                # cap.release () # Turn off the camera
        out.release()
