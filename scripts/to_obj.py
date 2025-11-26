from scripts import data_preparation as dp
import networkx as nx
def to_obj(vertices, metadata, edges, save_to, scale = 1.0):
    with open(save_to, 'w') as j:
        m = dict()
        for i, v in enumerate(vertices):
            m[v] = i + 1
            G = metadata[metadata['node_id'] == int(v)]
            x, y, z = G['x'].values[0], G['y'].values[0], G['z'].values[0]
            j.write(f'v {scale*x} {scale*y} {scale*z} #{v}\n')
        for e in edges:
            j.write(f'l {m[e[0]]} {m[e[1]]}\n')
    return m

neurons_ids = [
    "7055857",
    "1805418",
    "14260575",
    "5835799",
    "10160250",
    "7840203",
    "5019924",
    "13986477",
    "10167078",
    "7982896",
    "4119387",
    "17591442",
    "4227544",
    "10495502",
    "8069478",
    "3913629",
    "11279244",
    "16846805",
    "8980589",
    "3664102",
]

ctx = dp.simulation_context('Ilya/trash/delete_me_please', neurons_ids)

ctx.build_full_graph()
fg = nx.read_gml(ctx.path_to_full_graph)
metadata = ctx.node_metadata

#print()
#x, y, z = metadata['x'].to_numpy(), metadata['y'].to_numpy(), metadata['z'].to_numpy()

to_obj(fg.nodes, metadata, fg.edges, 'del.obj', 1/2000)