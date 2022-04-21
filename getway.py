import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append(".")
sys.path.append("model/")
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, History
from model.yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, yolo_eval
from model.yolo3.utils import get_random_data, letterbox_image
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import os, colorsys, configparser
import matplotlib.pyplot as plt
from datetime import datetime
class getway():
    def __init__(self,config_file='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)   
        self.model = None
        self.model_path=self.config.get("model","pre_train_model_path")
        self.train_loss_list = []
        self.test_loss_list = []
        self.log_dir = self.config.get("model","log_dir")
        self.classes_path = self.config.get("model","classes_path")
        self.anchors_path = self.config.get("model","anchors_path")
        self.input_shape=(self.config.getint("model","input_shape_1"),self.config.getint("model","input_shape_2"))
        self.sess = K.get_session()
        self.class_names = self.get_classes(self.classes_path)
        self.anchors=self.get_anchors(self.anchors_path)
        self.num_classes = len(self.class_names)
        self.gpu_num=self.config.getint("model","gpu_num")
        self.score=self.config.getfloat("model","score")
        self.iou=self.config.getfloat("model","iou")
        self.boxes, self.scores, self.classes = self.generate()

    def train_and_test(self, train_path, val_path, model_save_name=''):
        '''
        input
        train_path type filepath and txt format  ->  
        val_path type filepath and txt format -> 
        model_save_name: string
        ----
        train_loss -> save into self.train_loss_list 
        val_loss -> save into self.val_loss_list 
        '''
        anchors = self.get_anchors(self.anchors_path)
        if(model_save_name!=''):
            model_save_name = self.config.get("model","save_model_name2")

        self.model = self.create_model(load_pretrained=False,weights_path=self.model_path) # make sure you know what you freeze
        logging = TensorBoard(log_dir=self.log_dir)
        checkpoint = ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        with open(train_path) as f:
            train_lines = f.readlines()
        with open(val_path) as f:
            val_lines = f.readlines()

        num_val = int(len(val_lines))
        num_train = int(len(train_lines))

        # Train with frozen layers first, to get a stable loss.
        # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
        if True:
            self.model.compile(optimizer=Adam(lr=1e-3), loss={
                # use custom yolo_loss Lambda layer.
                'yolo_loss': lambda y_true, y_pred: y_pred})

            batch_size = self.config.getint("train","batch_size")
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            history = self.model.fit_generator(self.data_generator_wrapper(train_lines, batch_size, self.input_shape, anchors, self.num_classes),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=self.data_generator_wrapper(val_lines, batch_size, self.input_shape, anchors, self.num_classes),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=self.config.getint("train","epochs_frozen"),
                    initial_epoch=0,
                    callbacks=[logging, checkpoint])
            self.model.save_weights(self.log_dir + self.config.get("model","save_model_name1") + datetime.now().strftime("%Y%m%d_%H_%M_%S") + '.h5')
        # Unfreeze and continue training, to fine-tune.
        # Train longer if the result is not good.
        if True:
            for i in range(len(self.model.layers)):
                self.model.layers[i].trainable = True
            self.model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
            print('Unfreeze all of the layers.')

            batch_size = 32 # note that more GPU memory is required after unfreezing the body
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            history2 = self.model.fit_generator(self.data_generator_wrapper(train_lines, batch_size, self.input_shape, anchors, self.num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=self.data_generator_wrapper(val_lines, batch_size, self.input_shape, anchors, self.num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=self.config.getint("train","epochs_unfrozen"),
                initial_epoch=1,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
            
            self.model.save_weights(self.log_dir + model_save_name + datetime.now().strftime("%Y%m%d_%H_%M_%S") + '.h5')
        #self.train_loss_list = history.history['loss']
        #self.test_loss_list = history.history['val_loss'] 
        self.train_loss_list = history2.history['loss']
        self.test_loss_list = history2.history['val_loss'] 
        return  self.train_loss_list,  self.test_loss_list

    def batch_predict(self, image_datas):
        '''
        input
        image_datas type numpy 4 D array  [n, L, W, RGB]  
        output
            [boxes.numpy(), scores.numpy(), classes.numpy()]
        '''
        pred_boxes = []
        for i in range(len(image_datas)):
            image=image_datas[i]

            if self.input_shape != (None, None):
                assert self.input_shape[0]%32 == 0, 'Multiples of 32 required'
                assert self.input_shape[1]%32 == 0, 'Multiples of 32 required'
                boxed_image = letterbox_image(array_to_img(image), tuple(reversed(self.input_shape)))
            else:
                new_image_size = (image.shape[1] - (image.shape[1] % 32),
                                image.shape[0].height - (image.shape[0] % 32))
                boxed_image = letterbox_image(array_to_img(image), new_image_size)
            image = np.array(boxed_image, dtype='float32')

            image /= 255.
            image = np.expand_dims(image, 0)  # Add batch dimension.

            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.model.input: image,
                    self.input_image_shape: [image.shape[1], image.shape[2]]
                    #K.learning_phase(): 0
                })
            print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
            pred_boxes.append([out_boxes, out_scores, out_classes])
        return pred_boxes
    def predict(self, image_data):
        '''
        input
            image_data 4D array image_datas type numpy 4 D array  [n, L, W, RGB]
        output
            [boxes.numpy(), scores.numpy(), classes.numpy()]
        '''
        image=image_data[0]

        if self.input_shape != (None, None):
            assert self.input_shape[0]%32 == 0, 'Multiples of 32 required'
            assert self.input_shape[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(array_to_img(image), tuple(reversed(self.input_shape)))
        else:
            new_image_size = (image_data.shape[2] - (image_data.shape[2] % 32),
                              image_data.shape[1].height - (image_data.shape[1] % 32))
            boxed_image = letterbox_image(array_to_img(image), new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.model.input: image_data,
                self.input_image_shape: [image_data.shape[1], image_data.shape[2]]
                #K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        pred_bbox = [out_boxes, out_scores, out_classes]
        return pred_bbox
        
    def plot_loss(self):
        '''
        plot train and test loss
        '''
        if self.train_loss_list==[]:
            print("This model have not been trained.")
        else:
            plt.title("Train and Validate Loss")
            plt.ylabel("Epoches")
            plt.xlabel("Loss")

            plt.plot(self.train_loss_list, label="Train")
            plt.plot(self.test_loss_list, label="Validate")
            plt.legend(loc="upper left")
            plt.show()       
    def load_model(self, weight_path):
        '''
        input 
        Every framework could be different 
        ----
        model -> save into -> self.model 
        '''
    
        model_path = os.path.expanduser(weight_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.model = load_model(model_path, compile=False)
        except:
            self.model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(self.model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
                
        return self.model

    def data_generator_wrapper(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n==0 or batch_size<=0: return None
        return self.data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)
    def get_classes(self, classes_path):
        '''loads the classes'''
        with open(classes_path) as f:
            self.class_names = f.readlines()
        self.class_names = [c.strip() for c in self.class_names]
        return self.class_names

    def get_anchors(self, anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


    def create_model(self, load_pretrained=False, freeze_body=0, weights_path='model_data/yolo_weights.h5'):
        '''create the training model'''
        K.clear_session() # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w = self.input_shape
        num_anchors = len(self.anchors)

        print("num_anchors: ",num_anchors)

        y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
            num_anchors//3, self.num_classes+5)) for l in range(3)]

        model_body = yolo_body(image_input, num_anchors//3, self.num_classes)
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, self.num_classes))
        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze darknet53 body or freeze all but 3 output layers.
                num = (185, len(model_body.layers)-3)[freeze_body-1]
                for i in range(num): model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': self.anchors, 'num_classes': self.num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])
        self.model = Model([model_body.input, *y_true], model_loss)

        return self.model


    def data_generator(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if i==0:
                    np.random.shuffle(annotation_lines)
                image, box = get_random_data(annotation_lines[i], input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i = (i+1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size)
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.model = load_model(model_path, compile=False)
        except:
            self.model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes



