# matrix-completion-kit

欠測を含む2次元行列を、複数の低ランク行列分解法から選択して補間するための
汎用Pythonライブラリです。実装ファイルは `src/` 直下に配置されていますが、
インストール後の公開パッケージ名は `matrix_completion` です。

## インストール

```bash
python -m pip install -e .
```

必要な外部ライブラリは `numpy>=1.24`、対応Pythonは3.10以降です。

## 基本的な使用例

```python
import numpy as np
from matrix_completion import MatrixCompleter

x = np.array([
    [1.0, np.nan, 3.0],
    [2.0, 4.0, 6.0],
    [3.0, 6.0, np.nan],
])

model = MatrixCompleter(
    method="gaussian_als",
    rank=2,
    max_iter=300,
    tolerance=1e-6,
    random_state=42,
    preserve_observed=True,
    ridge=1e-3,
)
completed = model.fit_transform(x)
result = model.result_
```

`completed` は観測済み要素を維持し、欠測要素だけを推定値へ置換した行列です。
全要素に対する低ランクモデルの推定値は `result.reconstruction` にあります。

## `MatrixCompleter` の全コンストラクタ引数

```python
MatrixCompleter(
    method="gaussian_als",
    rank=2,
    *,
    max_iter=200,
    tolerance=1e-5,
    random_state=42,
    preserve_observed=True,
    ridge=1e-3,
    learning_rate=0.005,
    regularization=0.02,
    epsilon=1e-8,
    gamma_shape=1.0,
    gamma_rate=1.0,
)
```

| 引数 | 型・既定値 | 意味 |
|---|---|---|
| `method` | `str`, `"gaussian_als"` | 使用する分解手法。下記5手法または2つの互換エイリアスを指定 |
| `rank` | `int`, `2` | 潜在因子数。`1 <= rank <= min(matrix.shape)` |
| `max_iter` | `int`, `200` | 最大反復回数。1以上 |
| `tolerance` | `float`, `1e-5` | 目的関数の相対的収束判定値。0以上 |
| `random_state` | `int`等, `42` | Funk-SVDの初期値と更新順に使う乱数シード |
| `preserve_observed` | `bool`, `True` | 完成行列で観測済み要素を入力値のまま維持するか |
| `ridge` | `float`, `1e-3` | `gaussian_als`のL2正則化強度。0以上 |
| `learning_rate` | `float`, `0.005` | `funk_svd`のSGD学習率。0より大きい値 |
| `regularization` | `float`, `0.02` | `funk_svd`のL2正則化強度。0以上 |
| `epsilon` | `float`, `1e-8` | NMF系の数値安定化定数。0より大きい値 |
| `gamma_shape` | `float`, `1.0` | `gamma_poisson_nmf`のGamma shape。0より大きい値 |
| `gamma_rate` | `float`, `1.0` | `gamma_poisson_nmf`のGamma rate。0より大きい値 |

## 選択可能な手法と全固有オプション

### `gaussian_als`

行平均を切片として持つ、L2正則化付き交互最小二乗法です。負値を含む行列にも
使用できます。

| オプション | 既定値 | 意味 |
|---|---:|---|
| `ridge` | `1e-3` | 左右因子の線形方程式へ加えるL2正則化強度 |

旧名称 `bayesian_pca` も同じ手法を呼び出すエイリアスとして使用できます。ただし、
この実装はGaussian priorを持つMAP型点推定であり、事後サンプルや信用区間を返す
完全なBayesian PCAではありません。

```python
model = MatrixCompleter(method="gaussian_als", rank=5, ridge=1e-2)
```

### `funk_svd`

観測要素だけを使い、全体平均、行・列バイアス、左右潜在因子を確率的勾配法で
推定します。負値を含む行列にも使用できます。

| オプション | 既定値 | 意味 |
|---|---:|---|
| `learning_rate` | `0.005` | SGDの学習率 |
| `regularization` | `0.02` | バイアスと潜在因子に対するL2正則化強度 |

```python
model = MatrixCompleter(
    method="funk_svd",
    rank=5,
    learning_rate=0.003,
    regularization=0.05,
)
```

### `nmf_frobenius`

観測要素上のFrobenius損失（平均二乗誤差）を最小化する、マスク対応NMFです。
観測値はすべて0以上でなければなりません。

| オプション | 既定値 | 意味 |
|---|---:|---|
| `epsilon` | `1e-8` | ゼロ除算防止と因子の正値下限 |

```python
model = MatrixCompleter(method="nmf_frobenius", rank=5, epsilon=1e-9)
```

### `nmf_kl`

観測要素上の一般化Kullback–Leibler divergenceを最小化する、マスク対応NMFです。
観測値はすべて0以上でなければなりません。

| オプション | 既定値 | 意味 |
|---|---:|---|
| `epsilon` | `1e-8` | 対数・ゼロ除算防止と因子の正値下限 |

```python
model = MatrixCompleter(method="nmf_kl", rank=5, epsilon=1e-9)
```

### `gamma_poisson_nmf`

Gamma正則化を加えたPoisson/NMF型の座標更新による点推定です。観測値はすべて
0以上である必要があり、特にカウントまたは非負強度データを想定します。

| オプション | 既定値 | 意味 |
|---|---:|---|
| `epsilon` | `1e-8` | 対数・ゼロ除算防止と因子の正値下限 |
| `gamma_shape` | `1.0` | Gamma正則化のshapeパラメータ |
| `gamma_rate` | `1.0` | Gamma正則化のrateパラメータ |

旧名称 `bayesian_nmf` はこの手法のエイリアスです。完全なベイズ推論ではなく、
事後不確実性は出力しません。

```python
model = MatrixCompleter(
    method="gamma_poisson_nmf",
    rank=5,
    gamma_shape=1.0,
    gamma_rate=1.0,
    epsilon=1e-9,
)
```

## 入力行列とマスク

### NaNからマスクを自動作成

`observed` を省略すると、有限値を観測、`NaN`・`+inf`・`-inf` を欠測として扱います。

```python
completed = model.fit_transform(x)
```

### 明示的な品質マスク

欠測位置に有限のプレースホルダー値が入っている場合や、品質フラグで低品質値も除外
したい場合は、入力行列と同じ形状のBoolean配列を渡します。`True` が観測値、`False`
が補間対象です。`True` の位置には有限値が必要です。

```python
completed = model.fit_transform(
    matrix_with_placeholder,
    observed=quality_mask,
)
```

入力条件は以下のとおりです。

- `matrix` は空でない2次元配列
- `observed` を与える場合は `matrix.shape` と同一
- 少なくとも1要素は観測されている
- 観測扱いの要素はすべて有限値
- NMF系3手法では観測値がすべて0以上

ライブラリはNDVI用の定数加算、標準化、値域クリップを自動では行いません。これらの
用途固有変換は呼び出し側で実施し、補間後に逆変換してください。

## 全メソッド

### `fit(matrix, observed=None)`

指定手法を学習し、`self` を返します。学習後に `result_` が作成されます。

### `fit_transform(matrix, observed=None)`

`fit` を実行し、完成行列のコピーを返します。`preserve_observed=True` なら観測値は
元入力と完全に同じです。

## 学習後の全属性と戻り値

`model.result_` で得られる `CompletionResult` は次を持ちます。

| 属性 | 意味 |
|---|---|
| `reconstruction` | 全要素に対する低ランクモデルの推定行列 |
| `completed` | 欠測を補間し、設定に応じて観測値を維持した完成行列 |
| `left_factor` | 形状 `(行数, rank)` の左因子 |
| `right_factor` | 形状 `(rank, 列数)` の右因子 |
| `n_iter` | 実際に実行した反復回数 |
| `converged` | 収束判定を満たしたか |
| `objective` | 最終反復の観測要素上の目的関数値 |
| `method` | エイリアス解決後の正式手法名 |
| `metadata` | 手法固有情報を持つ読み取り専用mapping |

`metadata` の内容は手法ごとに異なります。

- `gaussian_als`: `row_mean`
- `funk_svd`: `global_mean`, `row_bias`, `column_bias`
- `nmf_frobenius`, `nmf_kl`: `divergence`
- `gamma_poisson_nmf`: `divergence`, `gamma_shape`, `gamma_rate`

`gaussian_als` の再構成には `row_mean[:, None] + left_factor @ right_factor`、
Funk-SVDにはmetadata内の全体・行・列バイアスも必要です。NMF系では
`left_factor @ right_factor` が再構成です。

## モデル選択上の注意

- rankは最終的な科学的結果とは独立に、人工マスクや交差検証などで選択してください。
- ランダムマスクは、実際の欠測が空間・時間的に集中する場合の精度を過大評価し得ます。
- 評価用マスクは、実運用で想定される欠測構造に近づけてください。
- `converged=False` の場合でも最終反復結果は返りますが、反復数、許容誤差、学習率、
  正則化強度を再検討してください。
