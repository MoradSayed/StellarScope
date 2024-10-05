def loadMesh(filename: str) -> list[float]:

    v = []
    vt = []
    vn = []

    vertices = []

    with open(filename, "r") as file:

        line = file.readline()

        while line:

            words = line.split(" ")
            match words[0]:
            
                case "v":
                    v.append(read_vertex_data(words))

                case "vt":
                    vt.append(read_texcoord_data(words))
                
                case "vn":
                    vn.append(read_normal_data(words))
            
                case "f":
                    read_face_data(words, v, vt, vn, vertices)
            
            line = file.readline()

    return vertices

def read_vertex_data(words: list[str]) -> list[float]:
    """
        Returns a vertex description.
    """

    return [
        float(words[1]),
        float(words[2]),
        float(words[3])
    ]
    
def read_texcoord_data(words: list[str]) -> list[float]:
    """
        Returns a texture coordinate description.
    """

    return [
        float(words[1]),
        float(words[2])
    ]
    
def read_normal_data(words: list[str]) -> list[float]:
    """
        Returns a normal vector description.
    """

    return [
        float(words[1]),
        float(words[2]),
        float(words[3])
    ]

def read_face_data(
    words: list[str], 
    v: list[list[float]], vt: list[list[float]], 
    vn: list[list[float]], vertices: list[float]) -> None:
    """
        Reads an edgetable and makes a face from it.
    """

    triangleCount = len(words) - 3

    for i in range(triangleCount):

        make_corner(words[1], v, vt, vn, vertices)
        make_corner(words[2 + i], v, vt, vn, vertices)
        make_corner(words[3 + i], v, vt, vn, vertices)

def make_corner(corner_description: str, 
    v: list[list[float]], vt: list[list[float]], 
    vn: list[list[float]], vertices: list[float]) -> None:
    """
        Composes a flattened description of a vertex.
    """

    v_vt_vn = corner_description.split("/")
    
    for element in v[int(v_vt_vn[0]) - 1]:
        vertices.append(element)
    for element in vt[int(v_vt_vn[1]) - 1]:
        vertices.append(element)
    for element in vn[int(v_vt_vn[2]) - 1]:
        vertices.append(element)


"""
if needed to save and load numpy arrays:
    np.save('test3.npy', a)    # .npy extension is added if not given
    d = np.load('test3.npy')
"""