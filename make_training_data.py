import os
pth = 'data/pascal_voc/VOC2012'
with open('splits/pascal_voc_train_aug.txt', 'r') as f:
	lines = [ line.rstrip('\n') for line in f.readlines() ]

with open( 'splits/pascal_voc_train_aug_image.txt', 'w') as f:
	for line in lines:
		f.write( pth + line.split(' ')[0] + '\n' )

with open( 'splits/pascal_voc_train_aug_label.txt', 'w') as f:
	for line in lines:
		f.write( pth + line.split(' ')[1] + '\n' )


with open('splits/pascal_voc_val.txt', 'r') as f:
	lines = [ line.rstrip('\n') for line in f.readlines() ]

with open( 'splits/pascal_voc_val_image.txt', 'w') as f:
	for line in lines:
		f.write( pth + line.split(' ')[0] + '\n' )

with open( 'splits/pascal_voc_val_label.txt', 'w') as f:
	for line in lines:
		f.write( pth + line.split(' ')[1] + '\n' )


