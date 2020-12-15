from tensorfuxx.train import *

class Session(object):

    def __init__(self, target='', graph=None, config=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, exec_tb):
        self.close()

    def run(self, fetches, feed_dict = {}, options=None, run_metadata=None):
        """Runs operations and evaluates tensors in `fetches`.
        This method runs one "step" of TensorFlow computation, by
        running the necessary graph fragment to execute every `Operation`
        and evaluate every `Tensor` in `fetches`, substituting the values in
        `feed_dict` for the corresponding input values.
        The `fetches` argument may be a single graph element, or an arbitrarily
        nested list, tuple, namedtuple, dict, or OrderedDict containing graph
        elements at its leaves.  A graph element can be one of the following types:
        * A `tf.Operation`.
          The corresponding fetched value will be `None`.
        * A `tf.Tensor`.
          The corresponding fetched value will be a numpy ndarray containing the
          value of that tensor.
        * A `tf.SparseTensor`.
          The corresponding fetched value will be a
          `tf.compat.v1.SparseTensorValue`
          containing the value of that sparse tensor.
        * A `get_tensor_handle` op.  The corresponding fetched value will be a
          numpy ndarray containing the handle of that tensor.
        * A `string` which is the name of a tensor or operation in the graph.
        The value returned by `run()` has the same shape as the `fetches` argument,
        where the leaves are replaced by the corresponding values returned by
        TensorFlow.
        Example:
        ```python
           a = tf.constant([10, 20])
           b = tf.constant([1.0, 2.0])
           # 'fetches' can be a singleton
           v = session.run(a)
           # v is the numpy array [10, 20]
           # 'fetches' can be a list.
           v = session.run([a, b])
           # v is a Python list with 2 numpy arrays: the 1-D array [10, 20] and the
           # 1-D array [1.0, 2.0]
           # 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
           MyData = collections.namedtuple('MyData', ['a', 'b'])
           v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
           # v is a dict with
           # v['k1'] is a MyData namedtuple with 'a' (the numpy array [10, 20]) and
           # 'b' (the numpy array [1.0, 2.0])
           # v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
           # [10, 20].
        ```
        The optional `feed_dict` argument allows the caller to override
        the value of tensors in the graph. Each key in `feed_dict` can be
        one of the following types:
        * If the key is a `tf.Tensor`, the
          value may be a Python scalar, string, list, or numpy ndarray
          that can be converted to the same `dtype` as that
          tensor. Additionally, if the key is a
          `tf.compat.v1.placeholder`, the shape of
          the value will be checked for compatibility with the placeholder.
        * If the key is a
          `tf.SparseTensor`,
          the value should be a
          `tf.compat.v1.SparseTensorValue`.
        * If the key is a nested tuple of `Tensor`s or `SparseTensor`s, the value
          should be a nested tuple with the same structure that maps to their
          corresponding values as above.
        Each value in `feed_dict` must be convertible to a numpy array of the dtype
        of the corresponding key.
        The optional `options` argument expects a [`RunOptions`] proto. The options
        allow controlling the behavior of this particular step (e.g. turning tracing
        on).
        The optional `run_metadata` argument expects a [`RunMetadata`] proto. When
        appropriate, the non-Tensor output of this step will be collected there. For
        example, when users turn on tracing in `options`, the profiled info will be
        collected into this argument and passed back.
        Args:
          fetches: A single graph element, a list of graph elements, or a dictionary
            whose values are graph elements or lists of graph elements (described
            above).
          feed_dict: A dictionary that maps graph elements to values (described
            above).
          options: A [`RunOptions`] protocol buffer
          run_metadata: A [`RunMetadata`] protocol buffer
        Returns:
          Either a single value if `fetches` is a single graph element, or
          a list of values if `fetches` is a list, or a dictionary with the
          same keys as `fetches` if that is a dictionary (described above).
          Order in which `fetches` operations are evaluated inside the call
          is undefined.
        Raises:
          RuntimeError: If this `Session` is in an invalid state (e.g. has been
            closed).
          TypeError: If `fetches` or `feed_dict` keys are of an inappropriate type.
          ValueError: If `fetches` or `feed_dict` keys are invalid or refer to a
            `Tensor` that doesn't exist.
        """
        mul = True
        if not isinstance(fetches, list):
            mul = False
            fetches = [fetches]
        self.executor = Executor(fetches)
        result = self.executor.run(feed_dict=feed_dict)
        if mul:
            return result
        else:
            return result[0]

    def close(self):
        """Closes this session.
        Calling this method frees all resources associated with the session.
        Raises:
          tf.errors.OpError: Or one of its subclasses if an error occurs while
            closing the TensorFlow session.
        """
        self._closed = True


zeros = np.zeros
ones = np.ones
float32 = np.float32
float64 = np.float64