import os
import laspy
import pandas as pd
import matplotlib.pyplot as plt
from util import las_to_pc, sexy_gif, rot_m

# parameters
outDir = 'out'
dataDir = 'MpalaForestGEO_LasClippedtoTreePolygons'
trees = [5302, 5021, 2537, 16338, 4205, 6559, 13093, 13481, 13749, 8255, 9666, 15930, 16236, 15120, 14976]

# make sure output directory exists
if not os.path.exists(outDir):
    os.makedirs(outDir)

# load metadata
metadata = pd.read_csv('labels.csv')

results = []
for id in trees:
    try:
        # load tree as pointcloud
        las = laspy.read(f'{dataDir}/treeID_{id}.las')
        pc, _ = las_to_pc(las)

        # create darkmode gif
        darkgifPath = f'{outDir}/treeID_{id}_dark.gif'
        sexy_gif(pc, darkgifPath, darkmode=True)
        
        # create lightmode gif
        lightgifPath = f'{outDir}/treeID_{id}_light.gif'
        sexy_gif(pc, lightgifPath, darkmode=False)

        # create flat image
        pc @= rot_m(0, -90, 0)
        pc -= pc.mean(dim=0)
        pc /= pc.norm(dim=-1).max()
        plt.axis('off')
        plt.scatter(pc[:,1], pc[:,0], c='k', s=1)
        flatPath = f'{outDir}/treeID_{id}_flat.png'
        plt.savefig(flatPath, dpi=1000)

        # save information
        results.append({
            'treeID': id,
            'species': metadata[metadata['tree_id'] == id]['label'].item(),
            'numPoints': len(las.classification),
            'darkGIF': darkgifPath,
            'lightGIF': lightgifPath,
            'flatImg': flatPath,
        })
    except Exception as e:
        print(f'Something went wrong with tree {id}: {e}')
        continue

pd.DataFrame.from_dict(results).to_csv('results.csv')