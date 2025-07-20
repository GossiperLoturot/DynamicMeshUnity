using UnityEngine;

public class SurfaceManager : MonoBehaviour
{
    [SerializeField]
    private int nodeSize = 16;
    [SerializeField]
    private int poolSize = 8;
    [SerializeField]
    private GameObject node;
    
    void Start()
    {
        for (var y = 0; y < poolSize; y++)
        for (var x = 0; x < poolSize; x++)
        {
            var position = new Vector3(x, 0, y) * nodeSize;
            var instance = Object.Instantiate(node, position, Quaternion.identity);
            instance.transform.SetParent(this.transform);
        }
    }
}
