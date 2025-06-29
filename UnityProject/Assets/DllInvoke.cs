using System.Runtime.InteropServices;
using UnityEngine;

public class DllInvoke : MonoBehaviour
{
    [DllImport("libdynmesh")]
    private static extern int naive_surface_nets(Vector3[] vertices, int[] triangles, float[] sdf, int size);
    
    void Start()
    {
        var size = 16;
        var vertices = new Vector3[size * size * size];
        var triangles = new int[size * size * size * 18];
        var sdf = new float[size * size * size];
        
        var origin = new Vector3(size / 2f, size / 2f, size / 2f);
        for (var x = 0; x < size; x++)
        {
            for (var y = 0; y < size; y++)
            {
                for (var z = 0; z < size; z++)
                {
                    var index = x + y * size + z * size * size;
                    var position = new Vector3(x, y, z);
                    sdf[index] = Vector3.Distance(position, origin) - 4f;
                }
            }
        }
        
        naive_surface_nets(vertices, triangles, sdf, size);

        var mesh = new Mesh();
        mesh.MarkDynamic();
        mesh.SetVertices(vertices);
        mesh.SetIndices(triangles, MeshTopology.Triangles, 0);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        GetComponent<MeshFilter>().sharedMesh = mesh;
    }
}
