//! What one llama.cpp context costs in memory.
//!
//! The weights are measured — `llama_model_size` reports the real byte count of
//! the loaded tensors, so nothing here has to guess from a quantization name.
//! The KV cache is not measurable before a context exists, but every input to
//! its size is known exactly, so it is computed from the standard llama.cpp
//! layout and labelled as a computation rather than a reading.

use koharu_llama::{context::params::KvCacheType, model::LlamaModel};

/// The size of one cache element, as a ggml block.
///
/// Quantized caches store 32 elements per block with shared scales, so a single
/// bytes-per-element number would have to be fractional. The block is kept whole
/// to stay in integer arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementSize {
    pub bytes_per_block: u32,
    pub elements_per_block: u32,
}

impl ElementSize {
    /// llama.cpp's default when no cache type is configured.
    pub const F16: Self = Self {
        bytes_per_block: 2,
        elements_per_block: 1,
    };

    /// The block layout of a type llama.cpp accepts for a KV cache.
    ///
    /// `None` for any other ggml type: llama.cpp rejects those when it creates
    /// a context, so there is no size to report rather than a size to guess.
    #[must_use]
    pub fn for_kv_cache_type(kind: KvCacheType) -> Option<Self> {
        let (bytes_per_block, elements_per_block) = match kind {
            KvCacheType::F32 => (4, 1),
            KvCacheType::F16 | KvCacheType::BF16 => (2, 1),
            KvCacheType::Q8_0 => (34, 32),
            KvCacheType::Q5_1 => (24, 32),
            KvCacheType::Q5_0 => (22, 32),
            KvCacheType::Q4_1 => (20, 32),
            KvCacheType::Q4_0 | KvCacheType::IQ4_NL => (18, 32),
            _ => return None,
        };
        Some(Self {
            bytes_per_block,
            elements_per_block,
        })
    }

    fn bytes_for(self, elements: u64) -> u64 {
        let per_block = u64::from(self.elements_per_block.max(1));
        // Round up: a partial block still occupies a whole one.
        elements
            .div_ceil(per_block)
            .saturating_mul(u64::from(self.bytes_per_block))
    }
}

/// The attention shape that decides how large a KV cache is.
///
/// `key_length` and `value_length` are the per-head dimensions. They are read
/// from GGUF metadata where the model states them, because `n_embd / n_head` is
/// only correct for architectures whose head dimension happens to divide the
/// embedding evenly — Gemma, for one, does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvGeometry {
    pub n_layer: u32,
    pub n_head_kv: u32,
    pub key_length: u32,
    pub value_length: u32,
}

impl KvGeometry {
    /// Reads the shape from a loaded model.
    ///
    /// Returns `None` when the head dimension cannot be established, rather than
    /// substituting a plausible number.
    pub fn read(model: &LlamaModel) -> Option<Self> {
        let architecture = model.meta_val_str("general.architecture").ok()?;
        let stated = |suffix: &str| {
            model
                .meta_val_str(&format!("{architecture}.attention.{suffix}"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
                .filter(|length| *length > 0)
        };
        let n_head = model.n_head();
        let derived = (n_head > 0)
            .then(|| u32::try_from(model.n_embd()).ok().map(|n_embd| n_embd / n_head))
            .flatten()
            .filter(|length| *length > 0);

        Some(Self {
            n_layer: model.n_layer(),
            n_head_kv: model.n_head_kv(),
            key_length: stated("key_length").or(derived)?,
            value_length: stated("value_length").or(derived)?,
        })
    }

    /// Bytes the key and value caches occupy at `n_ctx` positions.
    ///
    /// `n_layer × n_ctx × n_head_kv × head_length` elements per cache, sized by
    /// the configured cache type. This is llama.cpp's own layout; it excludes
    /// the compute graph and scratch buffers, which depend on batch shape and
    /// backend internals and are not estimable from here.
    #[must_use]
    pub fn cache_bytes(self, n_ctx: u32, key: ElementSize, value: ElementSize) -> u64 {
        let positions = u64::from(self.n_layer)
            .saturating_mul(u64::from(n_ctx))
            .saturating_mul(u64::from(self.n_head_kv));
        key.bytes_for(positions.saturating_mul(u64::from(self.key_length)))
            .saturating_add(value.bytes_for(positions.saturating_mul(u64::from(self.value_length))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q8_0: ElementSize = ElementSize {
        bytes_per_block: 34,
        elements_per_block: 32,
    };

    #[test]
    fn every_cache_type_the_settings_offer_has_a_block_layout() {
        for kind in [
            KvCacheType::F32,
            KvCacheType::F16,
            KvCacheType::BF16,
            KvCacheType::Q8_0,
            KvCacheType::Q5_1,
            KvCacheType::Q5_0,
            KvCacheType::Q4_1,
            KvCacheType::Q4_0,
            KvCacheType::IQ4_NL,
        ] {
            assert!(ElementSize::for_kv_cache_type(kind).is_some(), "{kind:?}");
        }
        assert_eq!(ElementSize::for_kv_cache_type(KvCacheType::Q4_K), None);
        assert_eq!(ElementSize::for_kv_cache_type(KvCacheType::F16), Some(ElementSize::F16));
    }

    /// Llama 3 8B: 32 layers, 8 KV heads, 128-wide heads.
    const LLAMA3_8B: KvGeometry = KvGeometry {
        n_layer: 32,
        n_head_kv: 8,
        key_length: 128,
        value_length: 128,
    };

    #[test]
    fn an_f16_cache_matches_the_hand_computed_size() {
        // 32 layers x 4096 positions x 8 heads x 128 wide x 2 bytes, twice.
        let expected = 2 * (32u64 * 4096 * 8 * 128 * 2);

        assert_eq!(
            LLAMA3_8B.cache_bytes(4096, ElementSize::F16, ElementSize::F16),
            expected
        );
        assert_eq!(expected, 512 * 1024 * 1024);
    }

    #[test]
    fn a_quantized_cache_is_sized_by_the_block_not_by_the_name() {
        let elements = 32u64 * 4096 * 8 * 128;
        let expected = 2 * (elements / 32 * 34);

        assert_eq!(LLAMA3_8B.cache_bytes(4096, Q8_0, Q8_0), expected);
    }

    #[test]
    fn key_and_value_types_are_sized_separately() {
        let mixed = LLAMA3_8B.cache_bytes(4096, ElementSize::F16, Q8_0);

        assert_eq!(
            mixed,
            (LLAMA3_8B.cache_bytes(4096, ElementSize::F16, ElementSize::F16)
                + LLAMA3_8B.cache_bytes(4096, Q8_0, Q8_0))
                / 2
        );
    }

    #[test]
    fn an_asymmetric_head_shape_is_not_assumed_square() {
        // DeepSeek-style MLA stores a wider key than value.
        let geometry = KvGeometry {
            key_length: 192,
            value_length: 128,
            ..LLAMA3_8B
        };

        assert_eq!(
            geometry.cache_bytes(1024, ElementSize::F16, ElementSize::F16),
            32 * 1024 * 8 * (192 + 128) * 2
        );
    }

    #[test]
    fn a_partial_block_still_occupies_a_whole_one() {
        let geometry = KvGeometry {
            n_layer: 1,
            n_head_kv: 1,
            key_length: 33,
            value_length: 33,
        };

        assert_eq!(geometry.cache_bytes(1, Q8_0, Q8_0), 2 * 2 * 34);
    }

    #[test]
    fn an_empty_context_costs_nothing() {
        assert_eq!(
            LLAMA3_8B.cache_bytes(0, ElementSize::F16, ElementSize::F16),
            0
        );
    }
}
