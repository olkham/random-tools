from threading import Thread
import cv2
import time
import os
import numpy as np
import requests

class FrameSource(object):

    @staticmethod
    def list_devices():
        """
        brute force test camera IDs
        https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
        """
        search = True
        device_id = 0
        working_devices = []
        available_devices = []
        while search:
            camera = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
            if not camera.isOpened():
                search = False
            else:
                is_reading, _ = camera.read()
                if is_reading:
                    working_devices.append(device_id)
                else:
                    available_devices.append(device_id)
            device_id +=1
        return available_devices, working_devices

    def __init__(self):
        self.default_frame = np.zeros((400,400,3), np.uint8)
        self.inference_frame = None
        self.running = False
        self.stopped = False
        self.skip = False
        self.source = None
        self.read_fail_count = 0
        
        self.folder_contents = []
        self.file_index = 0

        self.capture = cv2.VideoCapture()

    def connect(self, src, streaming=True):

        if str(src).endswith('m3u8'):
            #todo add youtube support
            result = requests.get(src)
            self.source_type = 'playlist'

        if src == self.source:
            #same source don't bother
            return

        if self.running:
            self.release()
            while not self.stopped:
                time.sleep(0.1)

        if type(src) == str:
            if os.path.isdir(src):
                self.folder_contents = os.listdir(src)
        else:
            self.folder_contents.clear()


        # Create a VideoCapture object
        if type(src) is int:
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            self.source_type = 'cam'
        elif len(self.folder_contents) == 0:
            self.capture = cv2.VideoCapture(src)
            self.source_type = 'vid'
        else:
            self.source_type = 'folder'

        # Create a VideoCapture object
        self.status, self.frame = None, None
        self.is_new_frame = False
        self.read_fail_count = 0
        self.read_count = 0
        self.loop_video = True
        self.streaming = streaming

        self.recording = False
        self.recording_index = 0
        self.output_video = None

        # get stream frame size
        self.prop_frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.prop_frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.prop_frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))-1

        self.last_frame_timestamp = time.time()
        self.fps = 0
        if self.source_type == 'vid':
            self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps_limit = 30
        self.source = src
        
        if self.streaming:
            self.start_stream()

    def isOpened(self):
        if self.source_type == 'folder':
            return True
        return self.capture.isOpened()

    def update(self):
        # Read the next frame from the stream in a different thread
        while self.running:
            if time.time() - self.last_frame_timestamp < 1/self.fps_limit:
                continue

            #TODO tidy up
            if len(self.folder_contents) > 0:
                self.frame = cv2.imread(os.path.join(self.source, self.folder_contents[self.file_index]))
                if self.frame is not None:
                    self.status = True
                    self.file_index = (self.file_index+1)%len(self.folder_contents)
                    self.is_new_frame = True

            if self.capture.isOpened() and not self.skip:
                try:
                    (self.status, self.frame) = self.capture.read()
                    if self.status:
                        self.read_count+=1

                        if self.loop_video:
                            if self.prop_frame_count > 0:
                                if self.read_count >= self.prop_frame_count:
                                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                    self.read_count = 0

                        elapsed = max(time.time() - self.last_frame_timestamp, 0.001)
                        self.fps = round(1/elapsed,1)
                        self.last_frame_timestamp = time.time()
                        self.is_new_frame = True
                    
                        if self.recording:
                            self.save_frame()
                except:
                    self.read_fail_count += 1
                    print(f'error count {self.read_fail_count}')

        print('------------------STOPPED------------------')
        self.stopped = True
        
    def read(self, wait=False):

        # pull frames on demand
        #TODO - remove duplicate code - this is the same in update
        if not self.streaming:

            if len(self.folder_contents) > 0:
                self.frame = cv2.imread(os.path.join(self.source, self.folder_contents[self.file_index]))
                if self.frame is not None:
                    self.status = True
                    self.file_index = (self.file_index+1)%len(self.folder_contents)
                    return True, self.frame

            if self.capture.isOpened() and not self.skip:
                (self.status, self.frame) = self.capture.read()
                if self.status:
                    self.read_count+=1

                    if self.loop_video:
                        if self.prop_frame_count > 0:
                            if self.read_count >= self.prop_frame_count:
                                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                self.read_count = 0

                    elapsed = max(time.time() - self.last_frame_timestamp, 0.001)
                    self.fps = round(1/elapsed,1)
                    self.last_frame_timestamp = time.time()
                    self.is_new_frame = True
                
                    if self.recording:
                        self.save_frame()


        if self.frame is None:
            return False, self.default_frame

        if wait:
            #hack
            while not self.is_new_frame:
                pass

        if self.is_new_frame:
            self.is_new_frame = False
            return True, self.frame
        return self.is_new_frame, self.frame

    def get_latest_frame_bytes(self):
        if self.frame is None:
            return False, cv2.imencode('.jpg', self.default_frame)[1].tobytes()
        if self.is_new_frame:
            self.is_new_frame = False
            return True, cv2.imencode('.jpg', self.frame)[1].tobytes()
        return self.is_new_frame, cv2.imencode('.jpg', self.frame)[1].tobytes()

    def stop_stream(self):
        self.capture.release()
        cv2.destroyAllWindows()
        if self.output_video is not None:
            self.output_video.release()
        exit(1)

    def stop(self):
        self.running = False

    def pause(self):
        self.skip = True

    def release(self):
        if self.running:
            self.running = False
        
        while not self.stopped:
            time.sleep(0.1)
        self.capture.release()
        self.source = None

    def show_frame(self, fps = 30):
        # Display frames in main program
        if self.status:
            cv2.namedWindow('Stream', 0)
            cv2.imshow('Stream', self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(round(1000/fps))
        if key == ord('q'):
            self.stop_stream()

        if key == ord('r'):
            self.record()

    def start_recording(self):
        # Save obtained frame into video output file
        if not self.recording:
            # Set up codec and output video settings
            self.codec = cv2.VideoWriter_fourcc(*'MJPG')
            self.output_video = cv2.VideoWriter(f'output_{self.recording_index}.avi', self.codec, 30, (self.frame_width, self.frame_height))
            if self.output_video.isOpened():
                self.recording = True

        self.save_frame()

    def save_frame(self):
        self.output_video.write(self.frame)

    def stop_recording(self):
        if self.recording:
            self.output_video.release()
            self.recording_index += 1
            self.recording = False

    def record(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_stream(self):

        if self.running and self.skip:
            self.skip = False
            return

        if not self.running:
            # Start the thread to read frames from the video stream
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = True
            self.running = True
            self.stopped = False
            self.thread.start()


if __name__ == '__main__':

    cv2.namedWindow('Stream', 0)

    frame_source = FrameSource()
    
    #cycle through some known resources
    sources = []
    sources.append(0)
    sources.append(2)
    sources.append('http://192.168.1.162:5004/auto/v1')
    sources.append('rtsp://admin:password@192.168.1.98:554/h264Preview_01_main')
    
    idx = 0
    frame_source.connect(sources[idx])
    run = True
    while run:
        try:
            latest, frame = frame_source.read(wait=True)
            if latest:
                cv2.imshow('Stream', frame)
                key = cv2.waitKey(2)

                if key == ord('q'):
                    run=False

                if key == ord('.'):
                    idx = (idx + 1) % len(sources)
                    frame_source.connect(sources[idx], True)

                if key == ord(','):
                    idx = (idx - 1) % len(sources)
                    frame_source.connect(sources[idx], True)
            
        except AttributeError:
            pass
