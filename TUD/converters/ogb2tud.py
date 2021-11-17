from ogb.graphproppred import GraphPropPredDataset
import tqdm as tq
import shutil
import os

def ogb2tu(dataname, out_dir='.', base = 1, attr_as_label=None, task=0, iszipped=False):
    dataset = GraphPropPredDataset(name = dataname)
    dirpath = os.path.join(out_dir,dataname)
    print("Working directory ",dirpath)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    f1 = open("%s/%s_A.txt"%(dirpath,dataname),'w')
    has_vertex_attr = True if 'node_feat' in dataset.graphs[0].keys() else False
    has_edge_attr = True if 'edge_feat' in dataset.graphs[0].keys() else False
    if has_vertex_attr: f2 = open("%s/%s_node_attributes.txt"%(dirpath,dataname),'w')
    if has_edge_attr: f3 = open("%s/%s_edge_attributes.txt"%(dirpath,dataname),'w')
    f4 = open("%s/%s_graph_labels.txt"%(dirpath,dataname),'w')
    f5 = open("%s/%s_graph_indicator.txt"%(dirpath,dataname),'w')
    has_vertex_label = True if (attr_as_label is not None and len(dataset.graphs[0]['node_feat'][0:]) > attr_as_label) else False
    if has_vertex_label: f6 = open("%s/%s_node_labels.txt"%(dirpath,dataname),'w')
    nonodes = 0
    y = dataset.labels
    for i,g in enumerate(tq.tqdm(dataset.graphs)):
        f4.write("%d\n"%(y[i][task]))
        j = 0
        for source,target in list(zip(g['edge_index'][0], g['edge_index'][1])):
            f1.write("%d, %d\n"%(source+base+(nonodes), target+base+(nonodes)))
            if has_edge_attr:
              edge_features = [str(e) for e in g['edge_feat'][j]]
              f3.write("%s\n"%(','.join(edge_features)))
            j += 1
        for node in range(g['num_nodes']):
            f5.write("%d\n"%(i+base))
            if has_vertex_attr:
              node_features = [str(n) for n in g['node_feat'][node,:]]
              f2.write("%s\n"%(','.join(node_features)))
            if has_vertex_label:
              try:
                f6.write("%d\n"%(g['node_feat'][node,attr_as_label]))
              except:
                raise Exception("Wrong attribute index as label")
            else:                        # otherwise, canculate degree and usei it as 'label'
              print("Warning: TUD format requires node labels!")

        nonodes += g['num_nodes']
    f1.close()
    if has_vertex_attr: f2.close()
    if has_edge_attr: f3.close()
    if has_vertex_label: f6.close()
    f4.close()
    f5.close()
    if iszipped:
        shutil.make_archive(os.path.join(out_dir,dataname), 'zip', dirpath)