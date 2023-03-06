from __future__ import annotations

from typing import List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np

from . import evaluate as dte


class NodeDict(TypedDict):
    type: Literal["node"]
    column: int
    value: float
    left: NodeDictType
    right: NodeDictType


class LeafDict(TypedDict):
    type: Literal["leaf"]
    label: str
    count: int


class Leaf:
    def __init__(
        self,
        label: str,
        depth: int,
        count: int,
        parent: Optional[Node] = None,
    ) -> None:
        self.label = label
        self.depth = depth
        self.count = count
        self.parent = parent

    def __call__(self, x: np.ndarray) -> str:
        return self.label


class Node:
    def __init__(
        self,
        column: int,
        value: float,
        depth: int,
        left: NodeType,
        right: NodeType,
        parent: Optional[Node] = None,
    ) -> None:
        self.column = column
        self.value = value
        self.depth = depth

        self.left = left
        self.right = right

        self.parent = parent

    def __repr__(self) -> str:
        return f"Node({self.column}, {self.value})"

    def __call__(self, x: np.ndarray) -> str:
        if x[self.column] < self.value:
            return self.left(x)
        else:
            return self.right(x)


NodeType = Union[Node, Leaf]
NodeDictType = Union[NodeDict, LeafDict]


class Tree:
    def __init__(self, algorithm: Literal["entropy", "gini"]) -> None:
        self.__root: Optional[NodeType] = None
        self.algorithm = (
            self.entropy if algorithm == "entropy" else self.negative_gini
        )

    @property
    def root(self) -> NodeType:
        if self.__root is None:
            raise ValueError("Tree has not been built.")

        return self.__root

    def __load(
        self,
        tree: NodeDictType,
        depth: int = 0,
        parent: Optional[Node] = None,
    ) -> NodeType:
        if tree["type"] not in ("leaf", "node"):
            raise SyntaxError(f"Type '{tree['type']}' not recognised.")
        elif tree["type"] == "leaf":
            return Leaf(tree["label"], depth, tree["count"], parent)

        left = self.__load(tree["left"], depth + 1)
        right = self.__load(tree["right"], depth + 1)

        node = Node(
            tree["column"],
            tree["value"],
            depth,
            left,
            right,
            parent,
        )

        left.parent = node
        right.parent = node

        return node

    def load(self, tree: NodeDictType) -> None:
        self.__root = self.__load(tree)

    def __dict(self, node: NodeType) -> NodeDictType:
        if isinstance(node, Leaf):
            return LeafDict(type="leaf", label=node.label, count=node.count)

        return NodeDict(
            type="node",
            column=node.column,
            value=node.value,
            left=self.__dict(node.left),
            right=self.__dict(node.right),
        )

    def dict(self) -> NodeDictType:
        if self.__root is None:
            raise ValueError("Tree has not been built.")

        return self.__dict(self.__root)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(a) for a in x])

    def predict_one(self, x: np.ndarray) -> str:
        return self.root(x)

    def __fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        depth: int = 0,
        parent: Optional[Node] = None,
    ) -> NodeType:
        if not x or not y:
            raise ValueError("Dataset or labels are empty.")
        elif x.shape[0] != y.shape[0]:
            raise ValueError("Dataset and label rows do not match.")
        elif y.shape[1] > 1:
            raise ValueError("Labels must be one dimensional.")

        labels, counts = np.unique(y, return_counts=True)

        if len(labels) == 1:
            return Leaf(labels[0], depth, counts[0], parent)

        column, value = self.best_split(x, y)

        left_x, right_x = self.split_dataset(x, x, column, value)
        left_y, right_y = self.split_dataset(x, y, column, value)

        left = self.__fit(left_x, left_y, depth + 1)
        right = self.__fit(right_x, right_y, depth + 1)

        node = Node(
            column,
            value,
            depth,
            left,
            right,
            parent,
        )

        left.parent = node
        right.parent = node

        return node

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.__root = self.__fit(x, y)

    def split_dataset(
        self, x: np.ndarray, data: np.ndarray, column: int, value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        left = data[x[:, column] < value]
        right = data[x[:, column] >= value]

        return left, right

    def best_split(self, x: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        best_split = None
        max_gain = 0

        for i, column in enumerate(x.T):
            for value in np.unique(column):
                split = i, column

                left, right = self.split_dataset(x, y, column, value)
                gain = self.information_gain(y, left, right)

                if gain > max_gain:
                    best_split = split
                    max_gain = gain

        if best_split is None:
            raise RuntimeError("Could not find a best split.")

        return best_split

    def information_gain(
        self, y: np.ndarray, left_split: np.ndarray, right_split: np.ndarray
    ) -> float:
        left = self.algorithm(left_split) * len(left_split) / len(y)
        right = self.algorithm(right_split) * len(right_split) / len(y)

        return self.algorithm(y) - (left + right)

    def negative_gini(self, y: np.ndarray) -> float:
        _, label_counts = np.unique(y, return_counts=True)
        p = label_counts / len(y)

        return -sum(p * (1 - p))

    def entropy(self, y: np.ndarray) -> float:
        _, label_counts = np.unique(y, return_counts=True)
        p = label_counts / len(y)

        return sum(-p * np.log2(p))

    def __tree_items(self, node: NodeType) -> List[NodeType]:
        if isinstance(node, Leaf):
            return [node]

        items = [node]

        return (
            items
            + self.__tree_items(node.left)
            + self.__tree_items(node.right)
        )

    def __majority_label(self, node: Node) -> Tuple[str, int]:
        if (
            isinstance(node, Leaf)
            or not isinstance(node.left, Leaf)
            or not isinstance(node.right, Leaf)
        ):
            raise ValueError("Must be a two-leaf node.")

        if node.left.count > node.right.count:
            return node.left.label, node.left.count

        return node.right.label, node.right.count

    def prune(self, x: np.ndarray, y: np.ndarray) -> None:
        tree_items = self.__tree_items(self.root)
        tree_items.sort(key=lambda x: x.depth, reverse=True)

        predictions = self.predict(x)
        results = dte.test_results(predictions, y)

        while tree_items:
            item = tree_items.pop(0)

            if (
                isinstance(item, Leaf)
                or not isinstance(item.left, Leaf)
                or not isinstance(item.right, Leaf)
            ):
                continue

            label, count = self.__majority_label(item)

            leaf = Leaf(label, item.depth, count, item.parent)

            if item.parent is None:
                if id(item) != self.root:
                    raise ValueError("Cannot reference tree without root.")
                self.__root = leaf
            elif id(item.parent.left) == id(item):
                item.parent.left = leaf
            else:
                item.parent.right = leaf

            prune_predictions = self.predict(x)
            prune_results = dte.test_results(prune_predictions, y)

            if prune_results.macro.f1 < results.macro.f1:
                if item.parent is None:
                    self.__root = item
                elif id(item.parent.left) == id(leaf):
                    item.parent.left = item
                else:
                    item.parent.right = item

                del leaf
