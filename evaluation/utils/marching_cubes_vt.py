from itertools import permutations

import numba
import numpy as np

import evaluation.utils.marching_cubes_lookup as lookup


def vertex_interpolate(p1, p2, v1, v2, isovalue):
    if np.any(p1 > p2):
        p1, p2, v1, v2 = p2, p1, v2, v1
    p = p1
    if np.abs(v1 - v2) > 1e-5:
        p = p1 + (p2 - p1) * (isovalue - v1) / (v2 - v1)
    return p


def extract_udfs_in_udf_combs(udf_combs):
    udfs = []
    for idx in idx_in_combs:
        udfs.append(udf_combs[idx[0], idx[1]])
    return udfs


inc = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
    ]
)

combs = []
comb_to_idx = [0] * 64
dist = [0] * 64
for i in range(7):
    for j in range(i + 1, 8):
        comb_to_idx[i * 8 + j] = len(combs)
        dist[i * 8 + j] = np.linalg.norm(inc[i] - inc[j])
        combs.append([i, j])

possible_assignments = []
for pos_num in range(0, 9):
    assignments = set(permutations([0] * (8 - pos_num) + [1] * pos_num))
    possible_assignments.extend([list(x) for x in assignments])

new_possible_assignments = []
for assignment in possible_assignments:
    if [1 - x for x in assignment] not in new_possible_assignments:
        new_possible_assignments.append(assignment)

possible_assignments = np.ascontiguousarray(np.array(new_possible_assignments))

idx_in_combs = [[0, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]]


def combs_to_verts(comb_values, udf=None):
    # comb_values.shape: 24
    max_comb_values = comb_values.max()
    if max_comb_values > 0.5:
        anchor_vert0, anchor_vert1 = combs[np.argmax(comb_values)]

        verts_class0 = [anchor_vert0]
        verts_class1 = [anchor_vert1]

        for temp_vert in range(8):
            if temp_vert == anchor_vert0 or temp_vert == anchor_vert1:
                continue
            temp_comb_value0 = comb_values[
                comb_to_idx[
                    min(temp_vert, anchor_vert0) * 8 + max(temp_vert, anchor_vert0)
                ]
            ]
            temp_comb_value1 = comb_values[
                comb_to_idx[
                    min(temp_vert, anchor_vert1) * 8 + max(temp_vert, anchor_vert1)
                ]
            ]
            if temp_comb_value0 > temp_comb_value1:
                verts_class1.append(temp_vert)
            else:
                verts_class0.append(temp_vert)

        if udf is None:
            result = np.zeros(8)
            for temp_vert in verts_class1:
                result[temp_vert] = 1
            return result
        else:
            result = np.ones(8) * -1
            for temp_vert in verts_class1:
                result[temp_vert] = 1
            vert_udf = np.array(extract_udfs_in_udf_combs(udf))
            result = result * vert_udf
            return result
    else:
        return np.zeros(8)


@numba.jit(nopython=True, fastmath=True)
def cal_loss(assignment, comb_values, udf, step_size):
    # inc = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])
    comb_to_idx = [0] * 64
    combs = []
    # dist = [0] * 64
    for i in range(7):
        for j in range(i + 1, 8):
            comb_to_idx[i * 8 + j] = len(combs)
            # dist[i * 8 + j] = np.sqrt(np.sum((inc[i] - inc[j]) ** 2))
            combs.append([i, j])

    neg_idxes = np.where(assignment < 0.5)[0]
    pos_idxes = np.where(assignment > 0.5)[0]

    gifs_loss = 0.0
    # udf_loss = 0.0
    for i in range(len(neg_idxes)):
        neg_idx = neg_idxes[i]
        for j in range(i + 1, len(neg_idxes)):
            gifs_loss = (
                gifs_loss +
                comb_values[
                    comb_to_idx[
                        min(neg_idx, neg_idxes[j]) * 8 + max(neg_idx, neg_idxes[j])
                    ],
                    0,
                ]
            )
        for pos_idx in pos_idxes:
            gifs_loss = (
                gifs_loss +
                1 -
                comb_values[
                    comb_to_idx[min(neg_idx, pos_idx) * 8 + max(neg_idx, pos_idx)], 0
                ]
            )
            # udf_item = udf[comb_to_idx[min(neg_idx, pos_idx) * 8 + max(neg_idx, pos_idx)]]
            # max_udf_diff = step_size / 2 * dist[comb_to_idx[min(neg_idx, pos_idx) * 8 + max(neg_idx, pos_idx)]]
            # udf_loss = udf_loss + max([np.abs(udf_item[0]) + np.abs(udf_item[1]), max_udf_diff]) - max_udf_diff

    for i in range(len(pos_idxes)):
        pos_idx = pos_idxes[i]
        for j in range(i + 1, len(pos_idxes)):
            gifs_loss += comb_values[
                comb_to_idx[
                    min(pos_idx, pos_idxes[j]) * 8 + max(pos_idx, pos_idxes[j])
                ],
                0,
            ]

    return gifs_loss


@numba.jit(nopython=True, fastmath=True)
def combs_to_verts_glb_opt(comb_values, udf, step_size, possible_assignments):
    idx_in_combs = [[0, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]]

    min_loss = np.inf
    min_idx = 0

    if np.max(comb_values) > 0.5:
        for assign_idx in range(128):
            assignment = possible_assignments[assign_idx]
            loss = cal_loss(assignment, comb_values, udf, step_size)
            if loss < min_loss:
                min_idx = assign_idx
                min_loss = loss

    result = np.ones(8) * -1
    result[possible_assignments[min_idx] > 0] = 1

    vert_udf = []
    for idx in idx_in_combs:
        vert_udf.append(udf[idx[0], idx[1]])
    vert_udf = np.array(vert_udf)

    result = result * vert_udf
    return result


def contrastive_marching_cubes(
    comb_values, isovalue=0.0, res=100, size=2.0, udf=None, selected_indices=None
):
    vs = {}
    fs = []
    mgrid = np.mgrid[: (res + 1), : (res + 1), : (res + 1)]
    mgrid = mgrid / res * size - size / 2
    mgrid = np.moveaxis(mgrid, 0, -1)

    if selected_indices is None:
        if udf is not None:
            udf = udf.reshape(res, res, res, len(combs), 2)
        comb_values = comb_values.reshape(res, res, res, len(combs))

        for step_x in range(res):
            for step_y in range(res):
                for step_z in range(res):
                    grid_inc = np.array([step_x, step_y, step_z]) + inc
                    grid_verts = mgrid[grid_inc[:, 0], grid_inc[:, 1], grid_inc[:, 2]]

                    temp_comb_values = comb_values[step_x, step_y, step_z]
                    if udf is not None:
                        temp_udf = udf[step_x, step_y, step_z]

                    if udf is None:
                        vert_values = combs_to_verts(temp_comb_values)
                    else:
                        vert_values = combs_to_verts(temp_comb_values, udf=temp_udf)

                    pow2 = 2 ** np.arange(8)
                    inside = (vert_values < isovalue).astype(np.int)
                    top_id = np.sum(inside * pow2)

                    edges = lookup.EDGE_TABLE[top_id]
                    if edges == 0:
                        continue

                    quick_lookup_key = np.packbits(
                        temp_comb_values < isovalue
                    ).tostring()

                    edge_cut = np.zeros((12, 3))
                    for i in range(12):
                        if edges & (1 << i):
                            p1, p2 = lookup.EDGE_VERTEX[i]
                            edge_cut[i] = vertex_interpolate(
                                grid_verts[p1],
                                grid_verts[p2],
                                vert_values[p1],
                                vert_values[p2],
                                isovalue,
                            )

                    tri_edges = lookup.TRI_TABLE[top_id] + [-1, -1]
                    tri_edges = [
                        tri_edges[3 * i: 3 * i + 3] for i in range(len(tri_edges) // 3)
                    ]
                    triangles = [edge_cut[e] for e in tri_edges if e[0] >= 0]
                    triangles = np.stack(triangles)

                    for t in triangles:
                        vid_list = []
                        for v in t:
                            v = tuple(v)
                            if v not in vs:
                                vs[v] = len(vs) + 1
                            vid_list.append(vs[v])
                        fs.append(vid_list)

    else:
        comb_values = comb_values.reshape(selected_indices.shape[0], len(combs), -1)
        udf = udf.reshape(selected_indices.shape[0], len(combs), -1)

        if udf is None:
            zip_datas = zip(comb_values, selected_indices)
        else:
            zip_datas = zip(comb_values, selected_indices, udf)
        for zip_data in zip_datas:
            if udf is None:
                temp_comb_values, selected_index = zip_data
            else:
                temp_comb_values, selected_index, temp_udf = zip_data
            grid_inc = selected_index + inc
            grid_verts = mgrid[grid_inc[:, 0], grid_inc[:, 1], grid_inc[:, 2]]

            if udf is None:
                vert_values = combs_to_verts(temp_comb_values)
            else:
                vert_values = combs_to_verts(temp_comb_values, temp_udf)
                # vert_values = combs_to_verts_glb_opt(
                #     temp_comb_values, temp_udf, size / res, possible_assignments
                # )

            pow2 = 2 ** np.arange(8)
            inside = (vert_values < isovalue).astype(np.int)
            top_id = np.sum(inside * pow2)

            edges = lookup.EDGE_TABLE[top_id]
            if edges == 0:
                continue

            edge_cut = np.zeros((12, 3))
            for i in range(12):
                if edges & (1 << i):
                    p1, p2 = lookup.EDGE_VERTEX[i]
                    edge_cut[i] = vertex_interpolate(
                        grid_verts[p1],
                        grid_verts[p2],
                        vert_values[p1],
                        vert_values[p2],
                        isovalue,
                    )

            tri_edges = lookup.TRI_TABLE[top_id] + [-1, -1]
            tri_edges = [
                tri_edges[3 * i: 3 * i + 3] for i in range(len(tri_edges) // 3)
            ]
            triangles = [edge_cut[e] for e in tri_edges if e[0] >= 0]
            triangles = np.stack(triangles)

            for t in triangles:
                vid_list = []
                for v in t:
                    v = tuple(v)
                    if v not in vs:
                        vs[v] = len(vs) + 1
                    vid_list.append(vs[v])
                fs.append(vid_list)

    return vs, fs


def get_grid_comb_verts(res=100, size=2.0, selected_indices=None, mgrid=None):
    # vertices range from [-size/2, -size/2, -size/2] to [size/2, size/2, size/2]
    if mgrid is None:
        mgrid = np.mgrid[: (res + 1), : (res + 1), : (res + 1)]
        mgrid = mgrid / res * size - size / 2
        mgrid = np.moveaxis(mgrid, 0, -1)

    grid_verts_inc = np.mgrid[:res, :res, :res]
    grid_verts_inc = np.moveaxis(grid_verts_inc, 0, -1)
    if selected_indices is None:
        grid_verts_inc = grid_verts_inc.reshape(-1, 3)
    else:
        grid_verts_inc = grid_verts_inc[
            selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]
        ]

    comb_inc0 = []
    comb_inc1 = []
    for comb in combs:
        comb_inc0.append(inc[comb[0]])
        comb_inc1.append(inc[comb[1]])
    comb_inc0 = np.array(comb_inc0)
    comb_inc1 = np.array(comb_inc1)

    comb_verts_inc0 = (
        np.repeat(grid_verts_inc[:, None, :], len(combs), axis=1) + comb_inc0[None]
    )
    comb_verts_inc1 = (
        np.repeat(grid_verts_inc[:, None, :], len(combs), axis=1) + comb_inc1[None]
    )
    comb_verts0 = mgrid[
        comb_verts_inc0[:, :, 0], comb_verts_inc0[:, :, 1], comb_verts_inc0[:, :, 2]
    ]
    comb_verts1 = mgrid[
        comb_verts_inc1[:, :, 0], comb_verts_inc1[:, :, 1], comb_verts_inc1[:, :, 2]
    ]

    if selected_indices is None:
        comb_grid_verts = (
            np.concatenate([comb_verts0, comb_verts1], axis=-1)
            .reshape(res, res, res, len(combs), 6)
            .copy()
        )
    else:
        comb_grid_verts = (
            np.concatenate([comb_verts0, comb_verts1], axis=-1)
            .reshape(-1, len(combs), 6)
            .copy()
        )
    return comb_grid_verts


def get_grid_comb_div(res=100, size=2.0, selected_indices=None, mgrid=None):
    # vertices range from [-size/2, -size/2, -size/2] to [size/2, size/2, size/2]
    if mgrid is None:
        mgrid = np.mgrid[: (res + 1), : (res + 1), : (res + 1)]
        mgrid = mgrid / res * size - size / 2
        mgrid = np.moveaxis(mgrid, 0, -1)

    grid_verts_inc = np.mgrid[:res, :res, :res]
    grid_verts_inc = np.moveaxis(grid_verts_inc, 0, -1)
    if selected_indices is None:
        grid_verts_inc = grid_verts_inc.reshape(-1, 3)
    else:
        grid_verts_inc = grid_verts_inc[
            selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]
        ]

    comb_inc0 = []
    comb_inc1 = []
    for comb in combs:
        comb_inc0.append(inc[comb[0]])
        comb_inc1.append(inc[comb[1]])
    comb_inc0 = np.array(comb_inc0)
    comb_inc1 = np.array(comb_inc1)

    comb_verts_inc0 = (
        np.repeat(grid_verts_inc[:, None, :], len(combs), axis=1) + comb_inc0[None]
    )
    comb_verts_inc1 = (
        np.repeat(grid_verts_inc[:, None, :], len(combs), axis=1) + comb_inc1[None]
    )
    comb_verts0 = mgrid[
        comb_verts_inc0[:, :, 0], comb_verts_inc0[:, :, 1], comb_verts_inc0[:, :, 2]
    ]
    comb_verts1 = mgrid[
        comb_verts_inc1[:, :, 0], comb_verts_inc1[:, :, 1], comb_verts_inc1[:, :, 2]
    ]

    if selected_indices is None:
        comb_grid_verts = (
            np.concatenate([comb_verts0, comb_verts1], axis=-1)
            .reshape(res, res, res, len(combs), -1)
            .copy()
        )
    else:
        comb_grid_verts = (
            np.concatenate([comb_verts0, comb_verts1], axis=-1)
            .reshape(selected_indices.shape[0], len(combs), -1)
            .copy()
        )
    return comb_grid_verts
