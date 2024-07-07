# UAS4_PengolahanCitra

# Image Segmentation with K-means Clustering

This project demonstrates image segmentation using the K-means clustering algorithm in Python. The code segments an image into different regions based on color similarity.

## Requirements

Ensure you have the following libraries installed:

- `numpy`
- `matplotlib`
- `opencv-python`

You can install these dependencies using pip:

```bash
pip install numpy matplotlib opencv-python
<div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-python"><span class="hljs-keyword">import</span> numpy <span class="hljs-keyword">as</span> np
<span class="hljs-keyword">import</span> matplotlib.pyplot <span class="hljs-keyword">as</span> plt
<span class="hljs-keyword">import</span> cv2

<span class="hljs-comment"># Memuat gambar sesuai dengan yang dimiliki</span>
image = cv2.imread(<span class="hljs-string">'Downloads/daun.jpg'</span>)
<span class="hljs-keyword">if</span> image <span class="hljs-keyword">is</span> <span class="hljs-literal">None</span>:
    <span class="hljs-keyword">raise</span> FileNotFoundError(<span class="hljs-string">"File gambar tidak ditemukan. Periksa kembali jalur file."</span>)

<span class="hljs-comment"># Mengubah warna gambar menjadi RGB (dari BGR)</span>
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

<span class="hljs-comment"># Membentuk ulang gambar menjadi susunan piksel 2D dengan 3 nilai warna (RGB)</span>
pixel_vals = image.reshape((-<span class="hljs-number">1</span>, <span class="hljs-number">3</span>))

<span class="hljs-comment"># Mengkonversikan tipe data ke float</span>
pixel_vals = np.float32(pixel_vals)

<span class="hljs-comment"># Menentukan kriteria agar algoritme berhenti berjalan: 100 iterasi atau epsilon 0.85</span>
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, <span class="hljs-number">100</span>, <span class="hljs-number">0.85</span>)

<span class="hljs-comment"># Menentukan jumlah cluster (K)</span>
k = <span class="hljs-number">3</span>

<span class="hljs-comment"># Melakukan k-means clustering</span>
retval, labels, centers = cv2.kmeans(pixel_vals, k, <span class="hljs-literal">None</span>, criteria, <span class="hljs-number">10</span>, cv2.KMEANS_RANDOM_CENTERS)

<span class="hljs-comment"># Mengkonversi data pusat cluster menjadi nilai 8-bit</span>
centers = np.uint8(centers)

<span class="hljs-comment"># Memetakan label ke warna pusat cluster</span>
segmented_data = centers[labels.flatten()]

<span class="hljs-comment"># Membentuk ulang data menjadi dimensi gambar asli</span>
segmented_image = segmented_data.reshape((image.shape))

<span class="hljs-comment"># Menampilkan gambar asli dan gambar tersegmentasi dalam satu plot</span>
plt.figure(figsize=(<span class="hljs-number">12</span>, <span class="hljs-number">6</span>))

plt.subplot(<span class="hljs-number">1</span>, <span class="hljs-number">2</span>, <span class="hljs-number">1</span>)
plt.imshow(image)
plt.title(<span class="hljs-string">'Gambar Asli'</span>)

plt.subplot(<span class="hljs-number">1</span>, <span class="hljs-number">2</span>, <span class="hljs-number">2</span>)
plt.imshow(segmented_image)
plt.title(<span class="hljs-string">'Gambar Tersegmentasi'</span>)

plt.show()

<span class="hljs-comment"># Menampilkan pusat cluster</span>
<span class="hljs-built_in">print</span>(<span class="hljs-string">"Pusat cluster:\n"</span>, centers)

<span class="hljs-comment"># Menampilkan distribusi label</span>
unique_labels, counts = np.unique(labels, return_counts=<span class="hljs-literal">True</span>)
<span class="hljs-built_in">print</span>(<span class="hljs-string">"Distribusi label:\n"</span>, <span class="hljs-built_in">dict</span>(<span class="hljs-built_in">zip</span>(unique_labels, counts)))
</code></div>
```
