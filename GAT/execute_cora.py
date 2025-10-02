import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

from models import GAT
from utils import process

checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

dataset = 'cora'

# training params
batch_size = 1
nb_epochs = 10000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8]  # numbers of hidden units per each attention head in each layer
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

adj = adj.todense()

features = features[np.newaxis]
adj = adj[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                             attn_drop, ffd_drop,
                             bias_mat=bias_in,
                             hid_units=hid_units, n_heads=n_heads,
                             residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    # 用于记录训练过程的指标
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    epochs_record = []

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                                                    feed_dict={
                                                        ftr_in: features[
                                                                tr_step * batch_size:(tr_step + 1) * batch_size],
                                                        bias_in: biases[
                                                                 tr_step * batch_size:(tr_step + 1) * batch_size],
                                                        lbl_in: y_train[
                                                                tr_step * batch_size:(tr_step + 1) * batch_size],
                                                        msk_in: train_mask[
                                                                tr_step * batch_size:(tr_step + 1) * batch_size],
                                                        is_train: True,
                                                        attn_drop: 0.6, ffd_drop: 0.6})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                                                 feed_dict={
                                                     ftr_in: features[vl_step * batch_size:(vl_step + 1) * batch_size],
                                                     bias_in: biases[vl_step * batch_size:(vl_step + 1) * batch_size],
                                                     lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                                                     msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
                                                     is_train: False,
                                                     attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            # 记录当前epoch的指标
            train_loss = train_loss_avg / tr_step
            train_acc = train_acc_avg / tr_step
            val_loss = val_loss_avg / vl_step
            val_acc = val_acc_avg / vl_step

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            epochs_record.append(epoch)

            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                  (train_loss, train_acc, val_loss, val_acc))

            if val_acc >= vacc_mx or val_loss <= vlss_mn:
                if val_acc >= vacc_mx and val_loss <= vlss_mn:
                    vacc_early_model = val_acc
                    vlss_early_model = val_loss
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc, vacc_mx))
                vlss_mn = np.min((val_loss, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        # 绘制训练过程中的损失和准确率曲线
        plt.figure(figsize=(14, 6))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs_record, train_losses, label='训练损失')
        plt.plot(epochs_record, val_losses, label='验证损失')
        plt.title('训练与验证损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs_record, train_accuracies, label='训练准确率')
        plt.plot(epochs_record, val_accuracies, label='验证准确率')
        plt.title('训练与验证准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('training_curves.png')  # 保存图表
        plt.show()  # 显示图表

        saver.restore(sess, checkpt_file)

        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                                             feed_dict={
                                                 ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                 bias_in: biases[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                 lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                 msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                 is_train: False,
                                                 attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)

        # 绘制测试结果柱状图
        plt.figure(figsize=(8, 6))
        plt.bar(['测试损失', '测试准确率'], [ts_loss / ts_step, ts_acc / ts_step], color=['#ff9999', '#66b3ff'])
        plt.title('测试集结果')
        plt.ylim(0, max(ts_loss / ts_step + 0.1, ts_acc / ts_step + 0.1))  # 设置y轴范围
        # 添加数值标签
        for i, v in enumerate([ts_loss / ts_step, ts_acc / ts_step]):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('test_results.png')
        plt.show()

        sess.close()