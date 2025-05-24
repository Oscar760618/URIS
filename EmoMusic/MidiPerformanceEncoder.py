# This part should be in environment of python 3.8
# Original code from magenta library
# ------------------------------------------------
'''
The Class for MidiPerformanceEncoderModel
'''
import note_seq
import pygtrie
import tempfile
from tensor2tensor.data_generators import text_encoder


class MidiPerformanceEncoder(object):
  """Convert between performance event indices and (filenames of) MIDI files."""

  def __init__(self, steps_per_second, num_velocity_bins, min_pitch, max_pitch,
               add_eos=False, ngrams=None):
    """Initialize a MidiPerformanceEncoder object.

    Encodes MIDI using a performance event encoding. Index 0 is unused as it is
    reserved for padding. Index 1 is unused unless `add_eos` is True, in which
    case it is appended to all encoded performances.

    If `ngrams` is specified, vocabulary is augmented with a set of n-grams over
    the original performance event vocabulary. When encoding, these n-grams will
    be replaced with new event indices. When decoding, the new indices will be
    expanded back into the original n-grams.

    No actual encoder interface is defined in Tensor2Tensor, but this class
    contains the same functions as TextEncoder, ImageEncoder, and AudioEncoder.

    Args:
      steps_per_second: Number of steps per second at which to quantize. Also
          used to determine number of time shift events (up to one second).
      num_velocity_bins: Number of quantized velocity bins to use.
      min_pitch: Minimum MIDI pitch to encode.
      max_pitch: Maximum MIDI pitch to encode (inclusive).
      add_eos: Whether or not to add an EOS event to the end of encoded
          performances.
      ngrams: Optional list of performance event n-grams (tuples) to be
          represented by new indices. N-grams must have length at least 2 and
          should be pre-offset by the number of reserved IDs.

    Raises:
      ValueError: If any n-gram has length less than 2, or contains one of the
          reserved IDs.
    """
    self._steps_per_second = steps_per_second
    self._num_velocity_bins = num_velocity_bins
    self._add_eos = add_eos
    self._ngrams = ngrams or []

    for ngram in self._ngrams:
      if len(ngram) < 2:
        raise ValueError('All n-grams must have length at least 2.')
      if any(i < self.num_reserved_ids for i in ngram):
        raise ValueError('N-grams cannot contain reserved IDs.')

    self._encoding = note_seq.PerformanceOneHotEncoding(
        num_velocity_bins=num_velocity_bins,
        max_shift_steps=steps_per_second,
        min_pitch=min_pitch,
        max_pitch=max_pitch)

    # Create a trie mapping n-grams to new indices.
    ngram_ids = range(self.unigram_vocab_size,
                      self.unigram_vocab_size + len(self._ngrams))
    self._ngrams_trie = pygtrie.Trie(zip(self._ngrams, ngram_ids))

    # Also add all unigrams to the trie.
    self._ngrams_trie.update(zip([(i,) for i in range(self.unigram_vocab_size)],
                                 range(self.unigram_vocab_size)))

  @property
  def num_reserved_ids(self):
    return text_encoder.NUM_RESERVED_TOKENS

  def encode_note_sequence(self, ns):
    """Transform a NoteSequence into a list of performance event indices.

    Args:
      ns: NoteSequence proto containing the performance to encode.

    Returns:
      ids: List of performance event indices.
    """
    performance = note_seq.Performance(
        note_seq.quantize_note_sequence_absolute(ns, self._steps_per_second),
        num_velocity_bins=self._num_velocity_bins)

    event_ids = [self._encoding.encode_event(event) + self.num_reserved_ids
                 for event in performance]

    # Greedily encode performance event n-grams as new indices.
    ids = []
    j = 0
    while j < len(event_ids):
      ngram = ()
      for i in range(j, len(event_ids)):
        ngram += (event_ids[i],)
        if self._ngrams_trie.has_key(ngram):
          best_ngram = ngram
        if not self._ngrams_trie.has_subtrie(ngram):
          break
      ids.append(self._ngrams_trie[best_ngram])
      j += len(best_ngram)

    if self._add_eos:
      ids.append(text_encoder.EOS_ID)

    return ids

  def encode(self, s):
    """Transform a MIDI filename into a list of performance event indices.

    Args:
      s: Path to the MIDI file.

    Returns:
      ids: List of performance event indices.
    """
    if s:
      ns = note_seq.midi_file_to_sequence_proto(s)
    else:
      ns = note_seq.NoteSequence()
    return self.encode_note_sequence(ns)

  def decode_to_note_sequence(self, ids, strip_extraneous=False):
    """Transform a sequence of event indices into a performance NoteSequence.

    Args:
      ids: List of performance event indices.
      strip_extraneous: Whether to strip EOS and padding from the end of `ids`.

    Returns:
      A NoteSequence.
    """
    if strip_extraneous:
      ids = text_encoder.strip_ids(ids, list(range(self.num_reserved_ids)))

    # Decode indices corresponding to event n-grams back into the n-grams.
    event_ids = []
    for i in ids:
      if i >= self.unigram_vocab_size:
        event_ids += self._ngrams[i - self.unigram_vocab_size]
      else:
        event_ids.append(i)

    performance = note_seq.Performance(
        quantized_sequence=None,
        steps_per_second=self._steps_per_second,
        num_velocity_bins=self._num_velocity_bins)
    for i in event_ids:
      performance.append(self._encoding.decode_event(i - self.num_reserved_ids))

    ns = performance.to_sequence()

    return ns

  def decode(self, ids, strip_extraneous=False):
    """Transform a sequence of event indices into a performance MIDI file.

    Args:
      ids: List of performance event indices.
      strip_extraneous: Whether to strip EOS and padding from the end of `ids`.

    Returns:
      Path to the temporary file where the MIDI was saved.
    """
    ns = self.decode_to_note_sequence(ids, strip_extraneous=strip_extraneous)

    _, tmp_file_path = tempfile.mkstemp('_decode.mid')
    note_seq.sequence_proto_to_midi_file(ns, tmp_file_path)

    return tmp_file_path

  def decode_list(self, ids):
    """Transform a sequence of event indices into a performance MIDI file.

    Args:
      ids: List of performance event indices.

    Returns:
      Single-element list containing path to the temporary file where the MIDI
      was saved.
    """
    return [self.decode(ids)]

  @property
  def unigram_vocab_size(self):
    return self._encoding.num_classes + self.num_reserved_ids

  @property
  def vocab_size(self):
    return self.unigram_vocab_size + len(self._ngrams)


