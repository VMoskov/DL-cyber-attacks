# Generate trigger
If you want to generate the trigger for BadNets, you should run one of the the commands below:</br><br/>
<b>CIFAR-10:</b><br/>
32x32 image with 3 color channels<br/><br/>
<b>MNIST:</b><br/>
28x28 image with 1 color channel<br/><br/>
```shell
python ./generate_square.py --image_size 32 --square_size 3 --distance_to_right 0 --distance_to_bottom 0 --color_channels 3 --output_path ./trigger_image_square.png
```

```shell
python ./generate_grid.py --image_size 32 --square_size 3 --distance_to_right 0 --distance_to_bottom 0 --color_channels 3 --output_path ./trigger_image_grid.png
```