# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import threading
import multiprocessing as mp

NUM_THREADS = 16
VIDEO_ROOT = '~/Datasets/Sth-sth/Sth-sth-v2-raw'         # Downloaded webm videos
FRAME_ROOT = '~/Datasets/Sth-sth/Sth-sth-v2-TSM-sliced'  # Directory for extracted frames


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video, tmpl='%06d.jpg'):
    # os.system(f'ffmpeg -i {VIDEO_ROOT}/{video} -vf -threads 1 -vf scale=-1:256 -q:v 0 '
    #           f'{FRAME_ROOT}/{video[:-5]}/{tmpl}')
    cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -vf scale=-1:256 -q:v 0 \"{}/{}/%06d.jpg\"'.format(VIDEO_ROOT, video,
                                                                                             FRAME_ROOT, video[:-5])
    os.system(cmd)


def target(video_list):
    for video in video_list:
        os.makedirs(os.path.join(FRAME_ROOT, video[:-5]))
        extract(video)


def extract_video(worker_id, task_queue):
    worker_id_str = "{0:05d}".format(worker_id)
    while True:
        task = task_queue.get()
        if ("DONE" == task):
            break
        # main job
        video = task
        os.makedirs(os.path.join(FRAME_ROOT, video[:-5]))
        extract(video)        
    


# multi-process wrapper
def extract_videos(video_list, num_proc):
    task_queue = mp.Queue()
    # init process
    process_list = []
    for _i in range(num_proc):
        p = mp.Process(target=extract_video, \
            args=(int(_i), task_queue))
        p.start()
        process_list.append(p)
    # init tasks
    tasks = video_list
    for _task in tasks:
        task_queue.put(_task)
    for i in range(num_proc):
        task_queue.put("DONE")
    # waiting for join
    for p in process_list:
        p.join()


if __name__ == '__main__':
    if not os.path.exists(VIDEO_ROOT):
        raise ValueError('Please download videos and set VIDEO_ROOT variable.')
    if not os.path.exists(FRAME_ROOT):
        os.makedirs(FRAME_ROOT)

    video_list = os.listdir(VIDEO_ROOT)
    
    # splits = list(split(video_list, NUM_THREADS))

    # threads = []
    # for i, split in enumerate(splits):
    #     thread = threading.Thread(target=target, args=(split,))
    #     thread.start()
    #     threads.append(thread)

    # for thread in threads:
    #     thread.join()

    extract_videos(video_list, NUM_THREADS)