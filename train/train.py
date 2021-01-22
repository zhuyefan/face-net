
# coding: utf-8

# In[1]:


import os
import random
import tensorflow as tf
import numpy as np
slim=tf.contrib.slim
import inception_resnet_v1 as network
import config
import sys
sys.path.append('../')
from align.utils import *


# In[ ]:


def main():
    image_size=(config.image_size,config.image_size)
    #创建graph和model存放目录
    if not os.path.exists(config.graph_dir):
        os.mkdir(config.graph_dir)
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    #获取图片地址和类别
    dataset=get_dataset(config.data_dir)
    #划分训练验证集
    if config.validation_set_split_ratio>0.0:
        train_set,val_set=split_dataset(dataset,config.validation_set_split_ratio,config.min_nrof_val_images_per_class)
    else:
        train_set,val_set=dataset,[]
    #训练集的种类数量
    nrof_classes=len(train_set)
    with tf.Graph().as_default():
        """
        
        global_step在滑动平均、优化器、指数衰减学习率等方面都有用到，
        这个变量的实际意义非常好理解：代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表
        """
        global_step=tf.Variable(0,trainable=False)
        #获取所有图像位置和相应类别
        image_list,label_list=get_image_paths_and_labels(train_set)
        assert len(image_list)>0, '训练集不能为空' # 若为空 触发异常输出 ‘训练集不能为空
        val_image_list,val_label_list=get_image_paths_and_labels(val_set)

        labels=tf.convert_to_tensor(label_list,dtype=tf.int32)


        """
        以上是对数据命名的处理，最后是 每个图片的路径对应一个lable 的形式
        
        """
        #样本数量
        range_size=labels.get_shape().as_list()[0]
        #每一个epoch的batch数量 每次epoch训练全部的数据 随机选择epoch_size个batch进行训练
        epoch_size=range_size//config.batch_size
        #创建一个队列，初始将所有数据进行入队
        index_queue=tf.train.range_input_producer(range_size,num_epochs=None,
                                                 shuffle=True,seed=None,capacity=32)
        # 避免出现不够一个batch的情况，将训练的数据输出队，作为训练数据
        index_dequeue_op=index_queue.dequeue_many(config.batch_size*epoch_size,'index_dequeue')

        batch_size_placeholder=tf.placeholder(tf.int32,name='batch_size')
        phase_train_placeholder=tf.placeholder(tf.bool,name='phase_train')
        image_paths_placeholder=tf.placeholder(tf.string,shape=(None,1),name='image_paths')
        labels_placeholder=tf.placeholder(tf.int32,shape=(None,1),name='label')
        keep_probability_placeholder=tf.placeholder(tf.float32,name='keep_probability')

        nrof_preprocess_threads=4
        #输入队列 最大可存储2000000个数据
        input_queue=tf.FIFOQueue(capacity=2000000,
                                 dtypes=[tf.string,tf.int32],
                                 shapes=[(1,),(1,)],
                                 shared_name=None,name=None)
        # 将形状为（none,1）该路径下的所有图片和标签 ，每每作为一个list入队，一个路径下一张图片
        enqueue_op=input_queue.enqueue_many([image_paths_placeholder,labels_placeholder],
                                           name='enqueue_op')
        #获取图像，label的batch形式，batch大小没有指定
        # 本来是有4组，每组都看成一个batch_size,但是若指定batch_size，则对组内进行删除，输出一组
        # 4个线程同时进行图片处理，并将原始图片进行裁剪，旋转，反转等操作进行数据增强，归一化加快图像处理
        # ，是在tensor_list中随机取一个，返回一个batch_size大小的数据（以每个标签为数据单位） batch是[某个人一张图片，标签]
        image_batch,label_batch=create_input_pipeline(input_queue,
                                                     image_size,
                                                     nrof_preprocess_threads,
                                                     batch_size_placeholder)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        # 网络输出 prelogits是模型网络后的输出， batch目前还是placeholder代替
        # 仍然是一张图片形成一个embedding,对应一个标签
        # 只是训练一个batch，但是该batch的类别无法确定，所以下面要进行全连接变成所有的类别长度
        prelogits,_=network.inference(image_batch,
                                      keep_probability_placeholder,
                                      phase_train=phase_train_placeholder,
                                      bottleneck_layer_size=config.embedding_size,
                                      weight_decay=config.weight_decay)
        # 用于计算loss 因为train_Set的长度和prelogits的长度不同
        # 将每张图片进行分类，以便与真实标签进行比较，所以要转换成相应shape
        logits=slim.fully_connected(prelogits,len(train_set),activation_fn=None,
                                   weights_initializer=slim.initializers.xavier_initializer(),
                                   weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                   scope='Logits', reuse=False)
        #正则化的embeddings主要用于测试，对比两张图片差异
        embeddings=tf.nn.l2_normalize(prelogits,1,1e-10,name='embeddings')
        #计算centerloss
        prelogits_center_loss,_=center_loss(prelogits,label_batch,config.center_loss_alfa,nrof_classes)
        tf.identity(prelogits_center_loss,name='center_loss')
        tf.summary.scalar('center_loss', prelogits_center_loss) # “”“输出包含单个标量值的'Summary'协议缓冲区。
        # 分段进行设置学习率，并将其添加进缓冲区
        boundaries = [int(epoch * range_size / config.batch_size) for epoch in config.LR_EPOCH]
        lr_values = [config.learning_rate * (0.1 ** x) for x in range(0, len(config.LR_EPOCH) + 1)]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, lr_values)
        tf.identity(learning_rate,name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        # 交叉熵损失
        # 将每张图片的真实标签与预测标签进行比较求交叉熵
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch,
                                                                     logits=logits,
                                                                     name='cross_entropy_per_example')
        # 求均值
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.identity(cross_entropy_mean,name='cross_entropy_mean')
        tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
        # l2正则loss（将 tensorflow自动维护一个tf.GraphKeys.WEIGHTS集合，
        # 手动在集合里面添加（tf.add_to_collection()）想要进行正则化惩罚的变量。）
        L2_loss=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        #总的loss
        total_loss=cross_entropy_mean+config.center_loss_factor*prelogits_center_loss+L2_loss
        tf.identity( total_loss,name='total_loss')
        tf.summary.scalar('total_loss', total_loss)

        # 将真实标签和预测标签进行equal,得到准确率
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        # 求其均值
        accuracy = tf.reduce_mean(correct_prediction)
        tf.identity(accuracy,name='accuracy')
        tf.summary.scalar('accuracy',accuracy)
        # global_step 会自动加一
        train_op=optimize(total_loss, global_step,
                        learning_rate,
                        config.moving_average_decay,
                        tf.global_variables())
        # 每次保留都会生成三个文件，
        # data文件是权重文件，index是一个索引文件，meta文件保留的图的结构
        # tf.trainable_variables()获取模型所有训练参数
        saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=3)
        summary_op=tf.summary.merge_all()
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        #训练和验证的graph保存地址
        """
        log是事件文件所在的目录，这里是工程目录下的log目录。第二个参数是事件文件要记录的图，也就是TensorFlow默认的图。
        可以调用其add_summary()方法将训练过程数据保存在filewriter指定的文件中
        """
        train_writer=tf.summary.FileWriter(config.graph_dir+'train/',sess.graph)
        val_writer = tf.summary.FileWriter(config.graph_dir+'val/', sess.graph)
        # 调用 tf.train.Coordinator() 来创建一个线程协调器，用来管理之后在Session中启动的所有线程;
        # 调用tf.train.start_queue_runners, 启动入队线程，由多个或单个线程，
        # 按照设定规则，把文件读入Filename Queue中。函数返回线程ID的列表，一般情况下，系统有多少个核，就会启动多少个入队线程（入队具体使用多少个线程在tf.train.batch中定义）
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if os.path.exists(config.model_dir):
                model_file=tf.train.latest_checkpoint(config.model_dir)
                if model_file:
                    saver.restore(sess,model_file)
                    print('重载模型训练')

            if not os.path.exists(config.model_dir):
                os.mkdir(config.model_dir)
            for epoch in range(1,config.max_nrof_epochs+1):
                step=sess.run(global_step,feed_dict=None)
                #训练
                batch_number = 0
                # 获取image和label
                # 得到每张图片的数据个标签 （符合满足整数batch的数据）
                index_epoch=sess.run(index_dequeue_op)
                label_epoch=np.array(label_list)[index_epoch]
                image_epoch=np.array(image_list)[index_epoch]

                labels_array = np.expand_dims(np.array(label_epoch),1)
                image_paths_array = np.expand_dims(np.array(image_epoch),1)
                #运行输入队列
                sess.run(enqueue_op,{image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
                while batch_number<epoch_size:
                    # 依次对每个batch进行损失函数，优化，网络传递训练等参数求解 （每个batch）
                    feed_dict = {phase_train_placeholder:True, batch_size_placeholder:config.batch_size,keep_probability_placeholder:config.keep_probability}
                    tensor_list = [total_loss, train_op, global_step,learning_rate, prelogits,
                                   cross_entropy_mean, accuracy, prelogits_center_loss]
                    #每经过100个batch更新一次graph
                    if batch_number % 100 == 0:
                        # 每训练100次batch。将第一百次的数据添加进图，并保存模型
                        loss_, _, step_, lr_,prelogits_, cross_entropy_mean_, accuracy_, center_loss_, summary_str = sess.run(tensor_list + [summary_op],  feed_dict=feed_dict)
                        train_writer.add_summary(summary_str, global_step=step_)
                        saver.save(sess=sess, save_path=config.model_dir+'model.ckpt',global_step=(step_))
                        print('epoch:%d/%d'%(epoch,config.max_nrof_epochs))

                        print("Step: %d/%d, accuracy: %3f, center loss: %4f, cross loss: %4f, Total Loss: %4f ,lr:%f " % (step_,epoch_size*config.max_nrof_epochs, accuracy_, center_loss_, cross_entropy_mean_,loss_, lr_))
                    else:
                        loss_, _, step_, lr_, prelogits_, cross_entropy_mean_, accuracy_, center_loss_,  = sess.run(tensor_list,  feed_dict=feed_dict)
                    batch_number+=1
                train_writer.add_summary(summary_str, global_step=step_)
                # 将验证集中的图片进行验证
                nrof_val_batches=len(val_label_list)//config.batch_size
                nrof_val_images=nrof_val_batches*config.batch_size

                labels_val_array=np.expand_dims(np.array(val_label_list[:nrof_val_images]),1)
                image_paths_val_array=np.expand_dims(np.array(val_image_list[:nrof_val_images]),1)
                #运行输入队列
                sess.run(enqueue_op, {image_paths_placeholder: image_paths_val_array, labels_placeholder: labels_val_array})
                loss_val_mean=0
                center_loss_val_mean=0
                cross_entropy_mean_val_mean=0
                accuracy_val_mean=0
                for i in range(nrof_val_batches):
                    feed_dict = {phase_train_placeholder:False, batch_size_placeholder:config.batch_size,keep_probability_placeholder:1.0}
                    loss_val,center_loss_val,cross_entropy_mean_val,accuracy_val,summary_val=sess.run ([total_loss,prelogits_center_loss,cross_entropy_mean, accuracy,summary_op], feed_dict=feed_dict)
                    loss_val_mean+=loss_val
                    center_loss_val_mean+=center_loss_val
                    cross_entropy_mean_val_mean+=cross_entropy_mean_val
                    accuracy_val_mean+=accuracy_val
                    if i % 10 == 9:
                        print('.', end='')
                        # waiting的作用。。。。。
                        sys.stdout.flush()
                # 计算每个batch的总损失值的均值和准确率的均值（90张图片的）
                val_writer.add_summary(summary_val, global_step=epoch)
                loss_val_mean/=nrof_val_batches
                center_loss_val_mean/=nrof_val_batches
                cross_entropy_mean_val_mean/=nrof_val_batches
                accuracy_val_mean/=nrof_val_batches
                print('到这了！')
                print("val: accuracy: %3f, center loss: %4f, cross loss: %4f, Total Loss: %4f " % ( accuracy_val_mean, center_loss_val_mean, cross_entropy_mean_val_mean,loss_val_mean))



# In[2]:


def center_loss(features,label,alfa,nrof_classes):
    '''计算centerloss
    参数：
      features:网络最终输出[batch,512]
      label:对应类别[batch,1]
      alfa:center更新比例
      nrof_classes:类别总数
    返回值：
      loss:center_loss损失值
      centers:中心点embeddings
    '''
    #embedding的维度
    nrof_features=features.get_shape()[1]
    centers=tf.get_variable('centers',[nrof_classes,nrof_features],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0),
                            trainable=False)
    label=tf.reshape(label,[-1])
    #挑选出每个batch对应的centers [batch,nrof_features]
    centers_batch=tf.gather(centers,label)
    diff=(1-alfa)*(centers_batch-features)
    #相同类别会累计相减
    centers=tf.scatter_sub(centers,label,diff)
    #先更新完centers在计算loss
    with tf.control_dependencies([centers]):
        loss=tf.reduce_mean(tf.square(features-centers_batch))
    return loss,centers


# In[3]:


def optimize(total_loss, global_step, learning_rate, moving_average_decay, update_gradient_vars):
    '''优化参数
    参数：
      total_loss:总损失函数
      global_step：全局step数
      learning_rate:学习率
      moving_average_decay：指数平均参数
      update_gradient_vars：需更新的参数
    返回值：
      train_op
    '''

    opt=tf.train.AdamOptimizer(learning_rate ,beta1=0.9, beta2=0.999, epsilon=0.1)
    #梯度计算
    grads=opt.compute_gradients(total_loss,update_gradient_vars)
    #应用更新的梯度
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    #参数和梯度分布图
    for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
    #指数平均
    variable_averages=tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op


# In[4]:


if __name__ == '__main__':
    main()
