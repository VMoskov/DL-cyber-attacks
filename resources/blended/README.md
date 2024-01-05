# Generate trigger
If you want to generate the trigger for Blended, you should run one of the the commands below:</br><br/>
<b>CIFAR-10:</b><br/>
32x32 image with 3 color channels<br/><br/>
<b>MNIST:</b><br/>
28x28 image with 1 color channel<br/><br/>
```shell
python ./generate_random_pattern.py --image_size 32 --color_channels 3 --output_path ./random_pattern_cifar10.png
```

```shell
python ./generate_random_pattern.py --image_size 28 --color_channels 1 --output_path ./random_pattern_mnist.png
```