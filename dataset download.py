import os
from datasets import load_dataset


def main():
	here = os.path.dirname(__file__)
	out_dir = os.path.join(here, "dataset")
	os.makedirs(out_dir, exist_ok=True)

	print("Loading ag_news dataset...")
	ds = load_dataset("ag_news")

	for split_name, split_ds in ds.items():
		out_path = os.path.join(out_dir, f"{split_name}.csv")
		print(f"Saving split '{split_name}' to {out_path}...")
		try:
			split_ds.to_csv(out_path, index=False)
		except Exception:
			# Fallback to pandas if direct to_csv isn't available
			split_ds.to_pandas().to_csv(out_path, index=False)

	print("Saved dataset splits to:")
	for p in sorted(os.listdir(out_dir)):
		print(" -", os.path.join(out_dir, p))


if __name__ == '__main__':
	main()