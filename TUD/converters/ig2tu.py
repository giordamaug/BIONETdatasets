import tqdm as tq
import os
import numpy as np
import shutil
def ig2tu(dataname, graphs, labels, up_to = None, out_dir = '.', base=1, iszipped=False,
          graph_attr = 'class', 
          remove_vertex_attr= ['id'], remove_edge_attr=[], 
          vertex_label_as_attr= False, edge_label_as_attr=False, 
          edge_weight_as_label=False,
          attr_as_label=None):
    vertex_attr = graphs[0].vertex_attributes()
    edge_attr = graphs[0].edge_attributes()
    is_weighted = True if graphs[0].is_weighted() else False
    has_vertex_label = True if 'label' in vertex_attr else False
    has_edge_label = True if 'label' in edge_attr else False
    has_graph_attr = True if graph_attr is not None and graph_attr in graphs[0].attributes() else False
    for attr in remove_vertex_attr: 
      if attr in vertex_attr:
        vertex_attr.remove(attr)
    for attr in remove_edge_attr : 
      if attr in edge_attr:
        edge_attr.remove(attr)
    if not vertex_label_as_attr and 'label' in vertex_attr: vertex_attr.remove('label')
    if not edge_label_as_attr and 'label' in edge_attr: edge_attr.remove('label')
    has_vertex_attr = True if len(vertex_attr) > 0 else False
    has_edge_attr = True if len(edge_attr) > 0 else False
    print('VERTEX ATTRIBUTES:', vertex_attr)
    print('EDGE ATTRIBUTES:', edge_attr)
    dirpath = os.path.join(out_dir,dataname)
    print("Working directory ",dirpath)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    f1 = open("%s/%s_A.txt"%(dirpath,dataname),'w')
    f2 = open("%s/%s_graph_indicator.txt"%(dirpath,dataname),'w')
    f3 = open("%s/%s_graph_labels.txt"%(dirpath,dataname),'w')
    f4 = open("%s/%s_node_labels.txt"%(dirpath,dataname),'w') 
    if has_vertex_attr: f5 = open("%s/%s_node_attributes.txt"%(dirpath,dataname),'w')
    if has_edge_label or (is_weighted and edge_weight_as_label): f6 = open("%s/%s_edge_labels.txt"%(dirpath,dataname),'w') 
    if has_edge_attr: f7 = open("%s/%s_edge_attributes.txt"%(dirpath,dataname),'w')
    if has_graph_attr: f8 = open("%s/%s_graph_attributes.txt"%(dirpath,dataname),'w')
    nonodes = 0
    for i,g in enumerate(tq.tqdm(graphs[:up_to])):
        f3.write("%d\n"%(labels[i]))
        if has_graph_attr: f8.write("%f\n"%(g[graph_attr]))
        for n,node in enumerate(g.vs):
            if has_vertex_attr: 
              comma_sep_attrlist = []
              for attr in vertex_attr:    # collect attributes values
                if type(node[attr]) == int or type(node[attr]) == float:
                  if attr in node.attribute_names():
                    comma_sep_attrlist += [str(node[attr])]
                  else:
                    comma_sep_attrlist += [str(0)]
                else:                     # only numeric attributes allowed
                  raise Exception('Cathegorical value not supported!')
              f5.write(', '.join(comma_sep_attrlist))
              f5.write('\n')
            if has_vertex_label:           # node has 'label'
              f4.write("%d\n"%(node['label'])) 
            else:                          # use 'attr_as_label' as 'label'
              if attr_as_label is not None and attr_as_label in vertex_attr and  type(attr_as_label) == str:
                f4.write("%d\n"%(node[attr_as_label]))
              else:                        # otherwise, canculate degree and usei it as 'label'
                if g.is_weighted(): f4.write("%d\n"%(g.strength(node, mode='all', loops=True, weights='weight')))
                else: f4.write("%d\n"%(g.degree([node])[0]))
            f2.write("%d\n"%(i+base))      # add graph indicator
            for edge in node.in_edges():
                if edge.target_vertex.index == node.index:
                    f1.write("%d, %d\n"%(edge.source_vertex.index+base+(nonodes), edge.target_vertex.index+base+(nonodes)))
        for n,edge in enumerate(g.es):
          if has_edge_label: f6.write("%d\n"%(edge['label']))
          else:
             if is_weighted and edge_weight_as_label:
               f6.write("%d\n"%(int(edge['weight'])))
          if has_edge_attr: 
              comma_sep_edgelist = []
              for attr in edge_attr:
                if type(edge[attr]) == int or type(edge[attr]) ==  float:
                  if attr in edge.attribute_names():
                    comma_sep_edgelist += [str(edge[attr])]
                  else:
                    comma_sep_edgelist += [str(0)]
                else:
                  raise Exception('Cathegorical value not supported!')
              f7.write(', '.join(comma_sep_edgelist))
              f7.write('\n')
        nonodes += g.vcount()
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    if has_vertex_attr: f5.close()
    if has_edge_label or (is_weighted and edge_weight_as_label): f6.close()
    if has_edge_attr: f7.close()
    if has_graph_attr: f8.close()
    if iszipped:
        shutil.make_archive(os.path.join(out_dir,dataname), 'zip', root_dir=out_dir, base_dir=dataname)

