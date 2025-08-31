using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using UnityEngine;

public class SurfaceNode : MonoBehaviour
{
    public class MeshData
    {
        public Vector3[] vertices;
        public Vector3[] normals;
        public Vector4[] tangents;
        public int[] triangles;
        public Bounds bounds;
    }
    
    private ConcurrentQueue<MeshData> _meshData;
    private MeshFilter _meshFilter;
    
    void Start()
    {
        _meshData = new ConcurrentQueue<MeshData>();
        _meshFilter = this.GetComponent<MeshFilter>();

        var origin = this.transform.position;
        Task.Run(() => Worker(origin));
    }
    
    private async Task Worker(Vector3 origin)
    {
        var size = 16 + 2;
        var sdf = new float[size * size * size];

        var vertices = new Vector3[size * size * size];
        var normals = new Vector3[size * size * size];
        var tangents = new Vector4[size * size * size];
        var triangles = new int[size * size * size * 18];
        var bounds = new Bounds[1];

        for (var x = 0; x < size; x++)
        for (var y = 0; y < size; y++)
        for (var z = 0; z < size; z++)
        {
            var index = x + y * size + z * size * size;
            var position = new Vector3(x, y, z) + origin;
            var height = Mathf.PerlinNoise(position.x * 0.1f, position.z * 0.1f) * 4f + 8f;
            sdf[index] = y - height;
        }

        naive_surface_nets(sdf, size, vertices, normals, tangents, triangles, bounds);
        
        var meshData = new MeshData
        {
            vertices = vertices,
            normals = normals,
            tangents = tangents,
            triangles = triangles,
            bounds = bounds[0]
        };
        _meshData.Enqueue(meshData);
    }
    
    private void Update()
    {
        if (_meshData.TryDequeue(out var meshData))
        {
            var mesh = new Mesh();
            
            mesh.MarkDynamic();
            mesh.SetVertices(meshData.vertices);
            mesh.SetNormals(meshData.normals);
            mesh.SetTangents(meshData.tangents);
            mesh.SetIndices(meshData.triangles, MeshTopology.Triangles, 0);
            mesh.bounds = meshData.bounds;
            
            _meshFilter.sharedMesh = mesh;
        }
    }

    [DllImport("libdynmesh")]
    private static extern int naive_surface_nets(
        float[] sdf, int size, Vector3[] vertices, Vector3[] normals, Vector4[] tangents, int[] triangles,
        Bounds[] bounds);
}