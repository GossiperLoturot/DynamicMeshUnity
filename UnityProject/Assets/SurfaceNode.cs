using System.Runtime.InteropServices;
using UnityEngine;

public class SurfaceNode : MonoBehaviour
{
    private void Start()
    {
        var size = 16 + 2;
        var vertices = new Vector3[size * size * size];
        var normals = new Vector3[size * size * size];
        var tangents = new Vector4[size * size * size];
        var triangles = new int[size * size * size * 18];
        var bounds = new Bounds[1];
        var sdf = new float[size * size * size];

        var origin = this.transform.position;
        for (var x = 0; x < size; x++)
        for (var y = 0; y < size; y++)
        for (var z = 0; z < size; z++)
        {
            var index = x + y * size + z * size * size;
            var position = new Vector3(x, y, z) + origin;
            var height = Mathf.PerlinNoise(position.x * 0.1f, position.z * 0.1f) * 4f + 8f;
            sdf[index] = y - height;
        }

        naive_surface_nets(vertices, normals, tangents, triangles, bounds, sdf, size);

        var mesh = new Mesh();
        mesh.MarkDynamic();
        mesh.SetVertices(vertices);
        mesh.SetNormals(normals);
        mesh.SetTangents(tangents);
        mesh.SetIndices(triangles, MeshTopology.Triangles, 0);
        mesh.bounds = bounds[0];
        GetComponent<MeshFilter>().sharedMesh = mesh;
    }

    [DllImport("libdynmesh")]
    private static extern int naive_surface_nets(
        Vector3[] vertices, Vector3[] normals, Vector4[] tangents, int[] triangles,
        Bounds[] bounds, float[] sdf, int size);
}
