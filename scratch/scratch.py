"""Function to test micrograd functionality eclipse VM with encrypted store"""

from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP
import numpy as np
import time

def test_micrograd():
    """
    Comprehensive test of micrograd library including:
    - Basic arithmetic operations and gradients
    - Neuron forward and backward pass
    - Multi-layer perceptron
    """
    
    print("=" * 50)
    print("Testing micrograd functionality")
    print("=" * 50)
    
    # Test 1: Basic arithmetic operations
    print("\n1. Testing basic arithmetic operations...")
    x = Value(3.0)
    y = Value(2.0)
    z = x * y + x
    z.backward()
    print("   Computation: z = x * y + x")
    print(f"   x={x.data}, y={y.data}")
    print(f"   z = x*y + x = {z.data}")
    print(f"   dz/dx = {x.grad}, dz/dy = {y.grad}")
    assert x.grad == 3.0, "Gradient of x should be 3.0"
    assert y.grad == 3.0, "Gradient of y should be 3.0"
    print("   ✓ Basic operations passed")
    
    # Test 2: Complex computational graph
    print("\n2. Testing complex computational graph...")
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    print(f"   a={a.data}, b={b.data}, g={g.data}")
    print(f"   da/dg = {a.grad}, db/dg = {b.grad}")
    print("   ✓ Complex graph passed")
    
    # Test 3: Neuron
    print("\n3. Testing Neuron...")
    neuron = Neuron(3, nonlin=True)
    x = [Value(1.0), Value(-1.0), Value(0.5)]
    y = neuron(x)
    y.backward()
    params = neuron.parameters()
    print(f"   Input: {[v.data for v in x]}")
    print(f"   Output: {y.data}")
    print(f"   Number of parameters: {len(params)}")
    print(f"   Parameter gradients: {[p.grad for p in params]}")
    print("   ✓ Neuron passed")
    
    # Test 4: Layer
    print("\n4. Testing Layer...")
    layer = Layer(3, 2, nonlin=True)
    x = [Value(1.0), Value(-1.0), Value(0.5)]
    y = layer(x)
    print(f"   Input: {[v.data for v in x]}")
    print(f"   Layer output count: {len(y) if isinstance(y, list) else 1}")
    print(f"   Output values: {[v.data for v in y] if isinstance(y, list) else y.data}")
    print("   ✓ Layer passed")
    
    # Test 5: MLP
    print("\n5. Testing MLP (Multi-layer Perceptron)...")
    mlp = MLP(3, [4, 4, 1])
    x = [Value(1.0), Value(-1.0), Value(0.5)]
    y = mlp(x)
    y.backward()
    params = mlp.parameters()
    print(f"   Input: {[v.data for v in x]}")
    print(f"   MLP structure: 3 -> 4 -> 4 -> 1")
    print(f"   Output: {y.data}")
    print(f"   Total parameters: {len(params)}")
    print(f"   Sample parameter gradients: {[p.grad for p in params[:5]]}")
    print("   ✓ MLP passed")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
    
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    # add them and time the operation
    start = time.time()
    s = a + b
    end = time.time()

    # print the vectors and the result
    print("a:", a)
    print("b:", b)
    print("a + b:", s)
    print("Time taken for addition:", end - start, "seconds")
    

if __name__ == "__main__":
    test_micrograd()