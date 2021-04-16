# Monolayout

## GAN model specific

### Generator
__Encoder__: ResnetEnc + Conv3x3_1 + max_pooling2D + Conv3x3_2 + max_pooling2D  
Параметры:  
* Input: batch of images (batch_size, 3, img_height, img_width)
* ResnetEnc
* Conv3x3_1: reflection padding(1), conv2D(out_channels=128, kernel_size=3)
* max_pooling2D: MaxPool2d(kernel_size=2)
* Conv3x3_2: reflection padding(1), conv2D(out_channels=128, kernel_size=3)
* Output: batch of encoded images (batch_size, 128, img_height/128, img_width/128)

__Decoder__: 5 blocks [Upconv_1 + BatchNorm + ReLU + Upsample + Upconv_2 + BatchNorm] + SoftMax + Conv3x3  
Параметры:
* Input: batch of encoded images (batch_size, 128, occ_map_size/2^5, occ_map_size/2^5)
* Upconv_1:  Conv2d(out_channels=[16, 32, 64, 128, 256], kernel_size=3, stride=1, padding=1)
* Upsample: interpolate(scale_factor=2, mode="nearest")
* Upconv_2:  Conv2d(out_channels=[16, 32, 64, 128, 256], kernel_size=3, stride=1, padding=1)
* SoftMax: disables on train (используется BCELoss, который содержит сигмоиду для стабильности)
* Conv3x3: reflection padding(1), conv2D(out_channels=2, kernel_size=3)
* Output: batch of output layouts (batch_size, 2, occ_map_size, occ_map_size)

### Discriminator

__model__: Conv3x3_1 + LeakyReLU + 3 blocks [Conv3x3_2 + BatchNorm + LeakyReLU] + Conv3x3_3 + Sigmoid  
Параметры:
* Input: batch of occ maps (batch_size, 2, occ_map_size, occ_map_size)
* Conv3x3_1: Conv2d(out_channels = 8, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
* LeakyReLU: negative_slope=0.2 
* Conv3x3_2: Conv2d(out_channels = [8, 32, 16], kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
* Conv3x3_3:Conv2d(out_channels = 1, kernel_size=3, stride=1, padding=1, bias=False)
* Output: patch output (batch_size, 1, occ_map_size/16, occ_map_size/16)


### Loss

__$topview\_loss$__: Crossentropy(generated_map, true_map)  
__$L_{GAN}$__: Binary Cross Entropy между предсказаниями дискриминатора на выходе генератора (fake_pred) и тензором из единиц (valid_tensor) на патчах размера (occ_map_size/16, occ_map_size/16) (насколько мы обманули дискриминатор)
__$L_D$__: Binary Cross Entropy между fake_pred и тензором из нулей на патчах (насколько хорошо распознали сгенерированные картинки) + Binary Cross Entropy между предсказаниями дискриминатора на реальных картах (real_pred) и valid_tensor на патчах (насколько хорошо распознали реальные картинки)  
__$L_G$__: $topview\_loss + \lambda \cdot L_{GAN}$, то есть насколько мы хорошо сгенерировали карту + регуляризатор $\cdot$ насколько мы обманули дискриминатор

### Train process
__Примечание__: Реальные картинки для модели на статике берутся из $OSM$, а для динамики используется $GT$ из обучения. 
* Бьем на батчи, прогоняем через Generator и считаем $L_{topview}$
* Прогоняем через дискриминатор полученные и фейковые карты, считаем на них ошибку распознавания $L_{GAN}$, ошибку дискриминатора $loss_D$ и ошибку генератора $loss_G$
* Спускаемся Adam по градиенту
    * Сначала по $loss_G$, потом по $loss_D$, если учим обе модели (после $5$ эпохи)
    * Иначе по $L_{topview}$, так как вклад дискриминатора не учитывается вообще.

## Non GAN model specific
### Structure:
__model__: состоит из Encoder + Decoder из предыдущей модели
### Loss
__topview loss__: Crossentropy(generated_map, true_map)  
### Train process

* Бьем на батчи, прогоняем через Generator и считаем $L_{topview}$
* Спускаемся Adam по градиенту $L_{topview}$


## Active changes

* Прикрутил флаги use_wandb и device для активации использования Weigth and Biases (логирую $loss_G$, $loss_D$, $topview\_loss$ после каждой эпохи и $mAP$, $IoU$ после $log\_freq$, а также выкидываю одну произвольную сгенерированную картинку с референсом раз в валидацию). Флаг get_onnx позволяет получить только .onnx файл. 
* Добавил класс набора данных KITTIRAWGT, состоящего только из размеченных элементов
* Добавил негенеративную модель
