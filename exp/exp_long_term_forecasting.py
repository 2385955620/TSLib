from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, print_gpu_memory
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import logging

warnings.filterwarnings('ignore')



class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # **************** #
    # 水下任务的组合损失函数
    def underwater_criterion(self, outputs, batch_y):
        f_dim=-2
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()
        logging.debug(f'outputs shape: {outputs.shape}, batch_y shape: {batch_y.shape}')
        # outputs: [batch_size, pred_len, 8]  batch_y: [batch_size, pred_len, 2]
        # 运动模式分类损失
        logging.debug(f'test marker')
        logging.debug(f'outputs slice shape: {outputs[:,:,0:4].reshape(-1,4).shape}, batch_y slice shape: {batch_y[:,:,0].long().reshape(-1).shape}')
        loss1 = criterion1(outputs[:,:,0:4].reshape(-1,4), batch_y[:,:,0].long().reshape(-1))
        loss2 = criterion2(outputs[:,:,4], batch_y[:,:,1])
        #logging.info(f'loss1 (classification loss): {loss1.item()}, loss2 (regression loss): {loss2.item()}')
        loss = loss1 + loss2
        #loss=loss2
        return loss


    def _select_criterion(self):
        if self.args.data=='UnderWater':
            criterion=self.underwater_criterion
            return criterion
        else:
            criterion = nn.MSELoss()
            return criterion
    
    

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # 增加注意力图处理
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]  # 
                            attns= outputs[1]  # 注意力图
                        else:
                            outputs = outputs
                            attns= None
                       
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # 增加注意力图处理
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # 
                        attns= outputs[1]  # 注意力图
                    else:
                        outputs = outputs
                        attns= None

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        # 提前结束训练配置
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        
        # if self.args.data=='UnderWater':   # 水下数据集的特殊损失函数
        #     criterion1, criterion2 = self._select_criterion()
        # else:
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        logging.info(f'Epochs: {self.args.train_epochs}, Train steps: {train_steps}, ')
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                logging.debug(f'Batch x shape: {batch_x.shape}, Batch y shape: {batch_y.shape}')
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # 增加注意力图处理
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]  # 
                            attns= outputs[1]  # 注意力图
                        else:
                            outputs = outputs
                            attns= None
                       

                        # if self.args.data=='UnderWater':
                        #     f_dim=-2
                        #     outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        #     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        #     loss1 = criterion1(outputs[:,:,0].reshape(-1), batch_y[:,:,0].long().reshape(-1))
                        #     loss2 = criterion2(outputs[:,:,1], batch_y[:,:,1])
                        #     loss = loss1 + loss2
                        #     train_loss.append(loss.item())
                        
                        # else:
                        # MS: 多变量预测单变量   M: 多变量预测单变量
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # 增加注意力图处理
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # 
                        attns= outputs[1]  # 注意力图
                    else:
                        outputs = outputs
                        attns= None

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)  # 内部保存checkpoint
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 打印 GPU 内存使用情况
            print_gpu_memory()

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model



    def test(self, setting, test=0):
        print('\n-----test-----')
        test_data, test_loader = self._get_data(flag='test')
        
        # 加载模型权重
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()

        from torchinfo import summary
        dummy_x_enc      = torch.randn(1, 120, 8).to(self.device)  # Encoder 输入
        dummy_x_mark_enc = torch.randn(1, 120, 6).to(self.device)  # Encoder 时间特征
        dummy_x_dec      = torch.randn(1, 61, 2).to(self.device)   # Decoder 输入
        dummy_x_mark_dec = torch.randn(1, 61, 6).to(self.device)   # Decoder 时间特征
        #summary(self.model, input_size=(1, self.args.seq_len, self.args.enc_in), device=str(self.device))
        summary(self.model, input_data=[dummy_x_enc,dummy_x_mark_enc,dummy_x_dec,dummy_x_mark_dec], device=str(self.device))
        # ===========================================================================
        # [新增] 模块：单样本推理时延测试 (Single Sample Inference Latency Benchmark)
        # ===========================================================================
        # 说明：这里只取第一个 batch 的第一个样本，重复推理 100 次来测算纯算力耗时
        # 不会影响后面正常的 preds/trues 结果统计
        print("\n[Benchmark] Starting Single Sample Inference Test...")
        
        try:
            # 1. 从 Loader 中手动取出一个 Batch
            iter_loader = iter(test_loader)
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter_loader)

            # 2. 切片取出第1个样本，并保持维度 [1, seq_len, features]
            # 注意：必须用 [0:1] 切片，不能用 [0]，否则会丢失 batch 维度导致报错
            one_sample_x = batch_x[0:1].float().to(self.device)
            one_sample_y = batch_y[0:1].float().to(self.device)
            one_sample_x_mark = batch_x_mark[0:1].float().to(self.device)
            one_sample_y_mark = batch_y_mark[0:1].float().to(self.device)

            # 3. 构造 Decoder 输入 (针对这一个样本)
            dec_inp = torch.zeros_like(one_sample_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([one_sample_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            # 4. 预热 (Warm-up) - 关键步骤
            # GPU 首次运行会有初始化开销，先空跑 10 次让其进入状态
            print("[Benchmark] Warming up GPU (10 iters)...")
            with torch.no_grad():
                for _ in range(10):
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            _ = self.model(one_sample_x, one_sample_x_mark, dec_inp, one_sample_y_mark)
                    else:
                        print(one_sample_x.shape, one_sample_x_mark.shape, dec_inp.shape, one_sample_y_mark.shape)

                        _ = self.model(one_sample_x, one_sample_x_mark, dec_inp, one_sample_y_mark)
            
            # 5. 正式测速循环 (运行 100 次取平均)
            test_interval = 100
            latency_list = []
            
            print(f"[Benchmark] Running {test_interval} iterations for stability...")
            with torch.no_grad():
                for _ in range(test_interval):
                    # --- 计时开始 ---
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize() # 等待 GPU 上一步操作完成
                    t_start = time.time()

                    # 模型推理
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            _ = self.model(one_sample_x, one_sample_x_mark, dec_inp, one_sample_y_mark)
                    else:
                        _ = self.model(one_sample_x, one_sample_x_mark, dec_inp, one_sample_y_mark)

                    # --- 计时结束 ---
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize() # 等待 GPU 计算完全结束
                    t_end = time.time()
                    
                    latency_list.append(t_end - t_start)

            # 6. 计算统计结果
            avg_latency = np.mean(latency_list)
            std_latency = np.std(latency_list)
            
            print("\n================ Single Sample Latency ================")
            print(f"Model Input Shape: {one_sample_x.shape}")
            print(f"Avg Latency: {avg_latency * 1000:.4f} ms")
            print(f"FPS:         {1.0 / avg_latency:.2f}")
            print("=======================================================\n")
            
        except StopIteration:
            print("[Benchmark] Warning: Test loader is empty, skipping benchmark.")
        except Exception as e:
            print(f"[Benchmark] Error during benchmark: {e}")



        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # 增加注意力图处理
                if isinstance(outputs, tuple):
                 
                    temp= outputs[0]  # 
                    attns= outputs[1]  # 注意力图
                    outputs = temp
                   
                    #attns=attns[0]
                    
                else:
                    outputs = outputs
                    attns= None
                #print(f'outputs shape: {outputs.shape}')
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                #logging.info(f'attens shape{attns.shape if attns is not None else "None"}')
                # 绘制图像
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    # 返归一化
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # 保存为pdf文件
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                
                
                # 绘制注意力图
                if i == 0 and attns is not None:
                    # 【修正 1】如果 attns 是 Tensor (单层输出)，手动把它变成列表
                    # 这样 enumerate 就会只循环 1 次，layer_idx=0
                    if isinstance(attns, torch.Tensor):
                        attns = [attns]
                        
                    for layer_idx, attn in enumerate(attns):
                        # attn shape: [batch_size, num_heads, d_var, d_var]
                        # 例如 (512, 8, 8, 8)    
                        # 【修正 2】取第 0 个样本，并在 Head 维度 (dim=0) 上做平均
                        # attn[0] -> shape [8, 8, 8] (Heads, Var, Var)
                        # .mean(dim=0) -> shape [8, 8] (Var, Var)
                        attn_map = attn[2].mean(dim=0).detach().cpu().numpy()
                        #attn_map = attn[2][1].detach().cpu().numpy()
                        print(f'Attention map shape (Layer {layer_idx+1}): {attn_map.shape}')
                        import matplotlib.pyplot as plt
                        import seaborn as sns # 推荐用 seaborn 画热力图更美观

                        plt.figure(figsize=(8, 6))
                        # 使用 seaborn 可以自动添加数值标注和更好的颜色
                        # x, y 轴标签可以设为变量名
                        sns.heatmap(attn_map, cmap='viridis', square=True) 
                        plt.title(f'Attention Map - Layer {layer_idx+1}')
                        plt.savefig(os.path.join(folder_path, f'attention_layer_{layer_idx+1}.pdf'))
                        plt.close()

        preds = np.concatenate(preds, axis=0) # 拼接batch结果
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)


        # 统计分类任务准确率
        if self.args.data=='UnderWater':
            from utils.tools import  cal_accuracy
            probs = torch.nn.functional.softmax(torch.tensor(preds[:,:,0:4]).reshape(-1, 4))  # (total_samples, num_classes) est. prob. for each class and sample
            #print(f'probs shape: {probs.shape}')
            predictions = torch.argmax(probs, dim=-1).cpu().numpy()  # (total_samples,) int class index for each sample
            class_trues = trues[:,:,0].flatten()
            #print(predictions.shape, class_trues.shape)
            #print(predictions[:10], class_trues[:10])
            accuracy = cal_accuracy(predictions, class_trues)
            print('Classification Accuracy: {:.4f}'.format(accuracy))
            



        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 计算 dtw 距离
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        logging.info(f'preds shape: {preds.shape}, trues shape: {trues.shape}')
        if self.args.data == 'UnderWater':
            mae, mse, rmse, mape, mspe = metric(preds[:,:,4], trues[:,:,1])
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
