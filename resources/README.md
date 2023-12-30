# Generate trigger
If you want to generate the trigger for BadNets, you should run one of the the commands below:
```shell
python badnets/generate_square.py --image_size 32 --square_size 3 --distance_to_right 0 --distance_to_bottom 0 --output_path ./badnets/trigger_image_square.png
```

```shell
python badnets/generate_grid.py --image_size 32 --square_size 3 --distance_to_right 0 --distance_to_bottom 0 --output_path ./badnets/trigger_image_grid.png
```