import numpy as np
from multiprocessing import Process, Manager, Lock
import time
import utils
import dataset
import pandas as pd

def collate_fn(batch):
    max_len=[]
    for param in range(len(batch[0])):
        max_len.append(max([i[param].shape[-1] for i in batch]))

    for i in range(len(batch)):
        batch_list = list(batch[i])
        for param in range(len(max_len)):
            batch_list[param] = np.pad(batch_list[param], ((0, 0),(0, max_len[param] - batch_list[param].shape[-1])), 'constant')
            # batch_list[param] = torch.nn.functional.pad(torch.tensor(batch_list[param]), (0, max_len[param] - batch_list[param].shape[-1]), 'constant', 0)
        batch[i] = tuple(batch_list)
    
    res=[]
    for param in range(len(batch[0])):
        res.append(np.stack([i[param] for i in batch]))
        # res.append(torch.stack((np.array([i[param] for i in batch]))))

    return tuple(res)

class BinaryDataLoader(object):
    def __init__(self,dataset_file_path='data/full_label/train',data_names=['input_feature','seg_target','edge_target'],batch_size=8,instances_buffer_size=3000, num_process=4,collate_fn=collate_fn):
        self.buffer_thread = None
        self.dataset_idx_path = dataset_file_path+'.idx'
        self.dataset_path = dataset_file_path+'.data'
        self.instances_buffer_size = instances_buffer_size
        self.batch_size = batch_size
        self.num_process = num_process
        self.buffer_single = Manager().list()
        self.buffer_batch = Manager().list()
        self.buffer_lock = Lock()
        self.data_names = data_names
        self.collate_fn=collate_fn

    def _fill_buf(self):
        if self.buffer_thread is None:
            self.buffer_thread = []
            for process in range(self.num_process):
                buffer_thread = Process(target=self.buf_thread, args=(process, self.num_process))
                buffer_thread.start()
                self.buffer_thread.append(buffer_thread)

    def _fill_batch(self):
        while len(self.buffer_single) > self.instances_buffer_size * 0.75:
            self.buffer_lock.acquire()
            num_data = len(self.buffer_single)
            batch_idx = np.random.choice(num_data, self.batch_size, replace=num_data < self.batch_size)
            batch_idx.sort()
            instances = [self.buffer_single.pop(i) for i in batch_idx[::-1]]
            self.buffer_lock.release()
            # for ins in instances:
            self.buffer_batch.append(self.collate_fn(instances))

    def buf_thread(self, process, num_process):
        print('=========start buf thread')
        read_count = 0
        while True:
            idx_data=pd.read_pickle(self.dataset_idx_path)
            data_file=open(self.dataset_path, "rb")
            num_data = len(idx_data)
            while True:
                #self.buffer_single装满啦，等等训练取走一些数据再装
                if len(self.buffer_single) >= self.instances_buffer_size:
                    max_batch_buffer = self.instances_buffer_size / self.batch_size#max(, 256)
                    if len(self.buffer_batch) < max_batch_buffer:
                        self._fill_batch()
                    else:
                        time.sleep(0.1)
                        continue
                #装到当前epcoh最后一个数据了
                read_count += 1
                if read_count >= num_data:
                    break
                #多个进程互不影响，确保不重复装数据
                if read_count % num_process != process:  # skip
                    continue
                #装一个数据
                data=[]
                for name in self.data_names:
                    data.append(utils.read_ndarray_from_bin(data_file,idx_data[name][read_count]))
                
                self.buffer_single.append(data)
            data_file.close()
            read_count = 0

    def __iter__(self):
        if self.buffer_thread is None:
            self._fill_buf()
        while True:
            if len(self.buffer_batch):
                yield self.buffer_batch.pop(0)


if __name__ == "__main__":
    dataloader=BinaryDataLoader()
    dataiter=iter(dataloader)
    while True:
        a,b,c=next(dataiter)
        print(a.shape,b.shape,c.shape)