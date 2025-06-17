using System.Runtime.InteropServices;
using UnityEngine;

public class DllInvoke : MonoBehaviour
{
    [DllImport("dynamic_mesh_lib")]
    private static extern int add_unity(int a, int b);
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Debug.Log(add_unity(10, 20));
    }
}
