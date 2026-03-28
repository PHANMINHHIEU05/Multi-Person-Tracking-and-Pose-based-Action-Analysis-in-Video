self.onmessage = async (e) => {
  // PERF: worker receives frame blobs
  if (e.data.type === "frame") {
    // PERF: decode only frame messages
    const bitmap = await createImageBitmap(e.data.blob); // PERF: off-main-thread decode
    self.postMessage({ type: "bitmap", bitmap }, [bitmap]); // PERF: transfer bitmap without copy
  }
};
