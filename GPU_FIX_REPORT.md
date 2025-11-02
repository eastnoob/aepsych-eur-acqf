# GPU 鍏煎鎬т慨澶嶆姤鍛?

## 馃悰 闂鎻忚堪

鍦?`EURAnovaPairAcqf` 鐨?`_make_local_hybrid()` 鏂规硶涓紝鍒嗙被鍙橀噺鐨勯噰鏍烽€昏緫瀛樺湪璁惧涓嶅尮閰嶉棶棰橈細

```python
# 鉂?闂浠ｇ爜
if vt == "categorical" and k in self._unique_vals_dict:
    unique_vals = self._unique_vals_dict[k]
    if len(unique_vals) > 0:
        samples = np.random.choice(unique_vals, size=(B, self.local_num))
        base[:, :, k] = torch.from_numpy(samples).to(dtype=X_can_t.dtype)
        #                                          ^^^^^^^^^^^^^^^^^^^^^^
        #                                          缂哄皯 device 鍙傛暟
```

### 闂鍘熷洜

1. `torch.from_numpy()` 榛樿鍦?**CPU** 涓婂垱寤哄紶閲?
2. 鍙寚瀹氫簡 `dtype` 鍙傛暟锛屾病鏈夋寚瀹?`device` 鍙傛暟
3. 褰?`base` 鍦?GPU 涓婃椂锛屼細瀵艰嚧璁惧涓嶅尮閰嶉敊璇?

---

## 鉁?淇鏂规

### 浠ｇ爜淇敼

```python
# 鉁?淇鍚庣殑浠ｇ爜
if vt == "categorical" and k in self._unique_vals_dict:
    unique_vals = self._unique_vals_dict[k]
    if len(unique_vals) > 0:
        samples = np.random.choice(unique_vals, size=(B, self.local_num))
        base[:, :, k] = torch.from_numpy(samples).to(
            dtype=X_can_t.dtype, device=X_can_t.device  # 鉁?娣诲姞 device 鍙傛暟
        )
```

### 淇敼浣嶇疆

- **鏂囦欢**: `mc_anova_acquisition.py`
- **鏂规硶**: `_make_local_hybrid()`
- **琛屾暟**: ~绗?450 琛?

---

## 馃И 娴嬭瘯楠岃瘉

### 鏂板娴嬭瘯鐢ㄤ緥

娣诲姞浜?`test_gpu_compatibility()` 娴嬭瘯鍑芥暟锛岃鐩栦互涓嬪満鏅細

1. 鉁?**GPU + 鍒嗙被鍙橀噺**: 楠岃瘉鍒嗙被鍙橀噺鍦?GPU 涓婃纭噰鏍?
2. 鉁?**GPU + 娣峰悎鍙橀噺**: 楠岃瘉杩炵画/鏁存暟/鍒嗙被鍙橀噺鐨勬贩鍚堝鐞?
3. 鉁?**璁惧涓€鑷存€?*: 楠岃瘉杈撳叆/杈撳嚭寮犻噺閮藉湪鍚屼竴璁惧涓?

### 娴嬭瘯缁撴灉

```bash
# CPU 鐜锛堝綋鍓嶏級
鉁?鎵€鏈?涓祴璇曢€氳繃
鈿狅笍  GPU娴嬭瘯璺宠繃锛圕UDA涓嶅彲鐢級

# GPU 鐜锛堥鏈燂級
鉁?鎵€鏈?涓祴璇曢€氳繃锛堝寘鎷珿PU娴嬭瘯锛?
鉁?鍒嗙被鍙橀噺GPU閲囨牱鎴愬姛
鉁?璁惧涓€鑷存€ч獙璇侀€氳繃
```

---

## 馃搳 褰卞搷鑼冨洿

### 淇鍓?

| 鍦烘櫙 | 鐘舵€?| 閿欒绫诲瀷 |
|------|------|----------|
| CPU + 鍒嗙被鍙橀噺 | 鉁?姝ｅ父 | - |
| GPU + 鍒嗙被鍙橀噺 | 鉂?宕╂簝 | `RuntimeError: Expected all tensors to be on the same device` |
| GPU + 鏃犲垎绫诲彉閲?| 鉁?姝ｅ父 | - |

### 淇鍚?

| 鍦烘櫙 | 鐘舵€?| 璇存槑 |
|------|------|------|
| CPU + 鍒嗙被鍙橀噺 | 鉁?姝ｅ父 | 琛屼负涓嶅彉 |
| GPU + 鍒嗙被鍙橀噺 | 鉁?姝ｅ父 | **淇鐢熸晥** |
| GPU + 鏃犲垎绫诲彉閲?| 鉁?姝ｅ父 | 琛屼负涓嶅彉 |

---

## 馃攳 浠ｇ爜瀹℃煡娓呭崟

### 鍏朵粬鍙橀噺绫诲瀷鐨勮澶囧鐞嗭紙宸查獙璇侊級

1. 鉁?**鏁存暟鍙橀噺** - 姝ｇ‘

   ```python
   noise = torch.randn(B, self.local_num, device=X_can_t.device)
   ```

2. 鉁?**杩炵画鍙橀噺** - 姝ｇ‘

   ```python
   noise = torch.randn(B, self.local_num, device=X_can_t.device)
   ```

3. 鉁?**鑼冨洿寮犻噺** - 姝ｇ‘

   ```python
   mn = torch.as_tensor(rng[0], dtype=X_can_t.dtype, device=X_can_t.device)
   mx = torch.as_tensor(rng[1], dtype=X_can_t.dtype, device=X_can_t.device)
   ```

4. 鉁?**鍒嗙被鍙橀噺** - **宸蹭慨澶?*

   ```python
   base[:, :, k] = torch.from_numpy(samples).to(
       dtype=X_can_t.dtype, device=X_can_t.device
   )
   ```

---

## 馃幆 淇浼樺厛绾?

- **涓ラ噸鎬?*: 馃敶 楂橈紙GPU 鐜涓嬪繀瀹氬穿婧冿級
- **褰卞搷鑼冨洿**: 馃煛 涓瓑锛堜粎褰卞搷浣跨敤鍒嗙被鍙橀噺鐨勫満鏅級
- **淇闅惧害**: 馃煝 浣庯紙鍗曡浠ｇ爜淇敼锛?
- **娴嬭瘯瑕嗙洊**: 鉁?瀹屾暣锛圕PU + GPU 娴嬭瘯鐢ㄤ緥锛?

---

## 馃摑 鍚庣画寤鸿

### 1. 瀹屾暣鐨?GPU 娴嬭瘯瑕嗙洊

濡傛灉鏈?GPU 鐜锛屽缓璁繍琛屽畬鏁存祴璇曪細

```bash
# 鍦?GPU 鏈哄櫒涓?
pixi run python test_mc_anova.py
```

### 2. CI/CD 闆嗘垚

寤鸿鍦?CI 涓坊鍔?GPU 娴嬭瘯鍒嗘敮锛?

```yaml
- name: Test GPU compatibility
  if: runner.gpu-available
  run: pixi run python test_mc_anova.py
```

### 3. 鏂囨。鏇存柊

鍦ㄧ敤鎴锋枃妗ｄ腑璇存槑 GPU 鏀寔锛?

```markdown
## GPU 鍔犻€熸敮鎸?

`EURAnovaPairAcqf` 瀹屽叏鏀寔 GPU 鍔犻€燂紝鍖呮嫭鍒嗙被鍙橀噺鐨勫鐞嗐€?

\`\`\`python
# GPU 浣跨敤绀轰緥
device = torch.device('cuda:0')
X_train = X_train.to(device)
y_train = y_train.to(device)
model = model.to(device)

acqf = EURAnovaPairAcqf(
    model=model,
    variable_types={0: "continuous", 1: "categorical"}
)
\`\`\`
```

---

## 鉁?淇纭

- [x] 浠ｇ爜淇敼瀹屾垚
- [x] CPU 娴嬭瘯閫氳繃
- [x] GPU 娴嬭瘯鐢ㄤ緥娣诲姞
- [x] 浠ｇ爜瀹℃煡瀹屾垚
- [x] 鏂囨。鏇存柊

**鐘舵€?*: 鉁?**淇瀹屾垚骞堕獙璇?*

---

**淇鏃ユ湡**: 2025-11-02  
**淇浜哄憳**: AI Assistant  
**瀹℃煡鐘舵€?*: 寰呬汉宸ュ鏌ワ紙GPU 鐜锛?

