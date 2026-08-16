// @__NO_SIDE_EFFECTS__
function nc(n) {
  const e = /* @__PURE__ */ Object.create(null);
  for (const t of n.split(",")) e[t] = 1;
  return (t) => t in e;
}
const ut = {}, ws = [], Fn = () => {
}, Bh = () => !1, Go = (n) => n.charCodeAt(0) === 111 && n.charCodeAt(1) === 110 && // uppercase letter
(n.charCodeAt(2) > 122 || n.charCodeAt(2) < 97), Wo = (n) => n.startsWith("onUpdate:"), Rt = Object.assign, ic = (n, e) => {
  const t = n.indexOf(e);
  t > -1 && n.splice(t, 1);
}, qd = Object.prototype.hasOwnProperty, it = (n, e) => qd.call(n, e), ze = Array.isArray, Rs = (n) => Pr(n) === "[object Map]", zh = (n) => Pr(n) === "[object Set]", Yc = (n) => Pr(n) === "[object Date]", Ye = (n) => typeof n == "function", xt = (n) => typeof n == "string", On = (n) => typeof n == "symbol", st = (n) => n !== null && typeof n == "object", Hh = (n) => (st(n) || Ye(n)) && Ye(n.then) && Ye(n.catch), Vh = Object.prototype.toString, Pr = (n) => Vh.call(n), jd = (n) => Pr(n).slice(8, -1), kh = (n) => Pr(n) === "[object Object]", sc = (n) => xt(n) && n !== "NaN" && n[0] !== "-" && "" + parseInt(n, 10) === n, ar = /* @__PURE__ */ nc(
  // the leading comma is intentional so empty string "" is also included
  ",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"
), Xo = (n) => {
  const e = /* @__PURE__ */ Object.create(null);
  return ((t) => e[t] || (e[t] = n(t)));
}, Kd = /-\w/g, Mn = Xo(
  (n) => n.replace(Kd, (e) => e.slice(1).toUpperCase())
), $d = /\B([A-Z])/g, $i = Xo(
  (n) => n.replace($d, "-$1").toLowerCase()
), Gh = Xo((n) => n.charAt(0).toUpperCase() + n.slice(1)), oa = Xo(
  (n) => n ? `on${Gh(n)}` : ""
), In = (n, e) => !Object.is(n, e), aa = (n, ...e) => {
  for (let t = 0; t < n.length; t++)
    n[t](...e);
}, Wh = (n, e, t, i = !1) => {
  Object.defineProperty(n, e, {
    configurable: !0,
    enumerable: !1,
    writable: i,
    value: t
  });
}, Zd = (n) => {
  const e = parseFloat(n);
  return isNaN(e) ? n : e;
}, Jd = (n) => {
  const e = xt(n) ? Number(n) : NaN;
  return isNaN(e) ? n : e;
};
let qc;
const Yo = () => qc || (qc = typeof globalThis < "u" ? globalThis : typeof self < "u" ? self : typeof window < "u" ? window : typeof global < "u" ? global : {});
function gi(n) {
  if (ze(n)) {
    const e = {};
    for (let t = 0; t < n.length; t++) {
      const i = n[t], s = xt(i) ? np(i) : gi(i);
      if (s)
        for (const r in s)
          e[r] = s[r];
    }
    return e;
  } else if (xt(n) || st(n))
    return n;
}
const Qd = /;(?![^(]*\))/g, ep = /:([^]+)/, tp = /\/\*[^]*?\*\//g;
function np(n) {
  const e = {};
  return n.replace(tp, "").split(Qd).forEach((t) => {
    if (t) {
      const i = t.split(ep);
      i.length > 1 && (e[i[0].trim()] = i[1].trim());
    }
  }), e;
}
function mr(n) {
  let e = "";
  if (xt(n))
    e = n;
  else if (ze(n))
    for (let t = 0; t < n.length; t++) {
      const i = mr(n[t]);
      i && (e += i + " ");
    }
  else if (st(n))
    for (const t in n)
      n[t] && (e += t + " ");
  return e.trim();
}
const ip = "itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly", sp = /* @__PURE__ */ nc(ip);
function Xh(n) {
  return !!n || n === "";
}
function rp(n, e) {
  if (n.length !== e.length) return !1;
  let t = !0;
  for (let i = 0; t && i < n.length; i++)
    t = rc(n[i], e[i]);
  return t;
}
function rc(n, e) {
  if (n === e) return !0;
  let t = Yc(n), i = Yc(e);
  if (t || i)
    return t && i ? n.getTime() === e.getTime() : !1;
  if (t = On(n), i = On(e), t || i)
    return n === e;
  if (t = ze(n), i = ze(e), t || i)
    return t && i ? rp(n, e) : !1;
  if (t = st(n), i = st(e), t || i) {
    if (!t || !i)
      return !1;
    const s = Object.keys(n).length, r = Object.keys(e).length;
    if (s !== r)
      return !1;
    for (const o in n) {
      const a = n.hasOwnProperty(o), l = e.hasOwnProperty(o);
      if (a && !l || !a && l || !rc(n[o], e[o]))
        return !1;
    }
  }
  return String(n) === String(e);
}
const Yh = (n) => !!(n && n.__v_isRef === !0), jt = (n) => xt(n) ? n : n == null ? "" : ze(n) || st(n) && (n.toString === Vh || !Ye(n.toString)) ? Yh(n) ? jt(n.value) : JSON.stringify(n, qh, 2) : String(n), qh = (n, e) => Yh(e) ? qh(n, e.value) : Rs(e) ? {
  [`Map(${e.size})`]: [...e.entries()].reduce(
    (t, [i, s], r) => (t[la(i, r) + " =>"] = s, t),
    {}
  )
} : zh(e) ? {
  [`Set(${e.size})`]: [...e.values()].map((t) => la(t))
} : On(e) ? la(e) : st(e) && !ze(e) && !kh(e) ? String(e) : e, la = (n, e = "") => {
  var t;
  return (
    // Symbol.description in es2019+ so we need to cast here to pass
    // the lib: es2016 check
    On(n) ? `Symbol(${(t = n.description) != null ? t : e})` : n
  );
};
let Ct;
class op {
  // TODO isolatedDeclarations "__v_skip"
  constructor(e = !1) {
    this.detached = e, this._active = !0, this._on = 0, this.effects = [], this.cleanups = [], this._isPaused = !1, this._warnOnRun = !0, this.__v_skip = !0, !e && Ct && (Ct.active ? (this.parent = Ct, this.index = (Ct.scopes || (Ct.scopes = [])).push(
      this
    ) - 1) : (this._active = !1, this._warnOnRun = !1));
  }
  get active() {
    return this._active;
  }
  pause() {
    if (this._active) {
      this._isPaused = !0;
      let e, t;
      if (this.scopes) {
        const i = this.scopes.slice();
        for (e = 0, t = i.length; e < t; e++)
          i[e].pause();
      }
      for (e = 0, t = this.effects.length; e < t; e++)
        this.effects[e].pause();
    }
  }
  /**
   * Resumes the effect scope, including all child scopes and effects.
   */
  resume() {
    if (this._active && this._isPaused) {
      this._isPaused = !1;
      let e, t;
      if (this.scopes) {
        const s = this.scopes.slice();
        for (e = 0, t = s.length; e < t; e++)
          s[e].resume();
      }
      const i = this.effects.slice();
      for (e = 0, t = i.length; e < t; e++)
        i[e].resume();
    }
  }
  run(e) {
    if (this._active) {
      const t = Ct;
      try {
        return Ct = this, e();
      } finally {
        Ct = t;
      }
    }
  }
  /**
   * This should only be called on non-detached scopes
   * @internal
   */
  on() {
    ++this._on === 1 && (this.prevScope = Ct, Ct = this);
  }
  /**
   * This should only be called on non-detached scopes
   * @internal
   */
  off() {
    if (this._on > 0 && --this._on === 0) {
      if (Ct === this)
        Ct = this.prevScope;
      else {
        let e = Ct;
        for (; e; ) {
          if (e.prevScope === this) {
            e.prevScope = this.prevScope;
            break;
          }
          e = e.prevScope;
        }
      }
      this.prevScope = void 0;
    }
  }
  stop(e) {
    if (this._active) {
      this._active = !1;
      let t, i;
      for (t = 0, i = this.effects.length; t < i; t++)
        this.effects[t].stop();
      for (this.effects.length = 0, t = 0, i = this.cleanups.length; t < i; t++)
        this.cleanups[t]();
      if (this.cleanups.length = 0, this.scopes) {
        const s = this.scopes.slice();
        for (t = 0, i = s.length; t < i; t++)
          s[t].stop(!0);
        this.scopes.length = 0;
      }
      if (!this.detached && this.parent && !e) {
        const s = this.parent.scopes.pop();
        s && s !== this && (this.parent.scopes[this.index] = s, s.index = this.index);
      }
      this.parent = void 0;
    }
  }
}
function ap() {
  return Ct;
}
let ft;
const ca = /* @__PURE__ */ new WeakSet();
class jh {
  constructor(e) {
    this.fn = e, this.deps = void 0, this.depsTail = void 0, this.flags = 5, this.next = void 0, this.cleanup = void 0, this.scheduler = void 0, Ct && (Ct.active ? Ct.effects.push(this) : this.flags &= -2);
  }
  pause() {
    this.flags |= 64;
  }
  resume() {
    this.flags & 64 && (this.flags &= -65, ca.has(this) && (ca.delete(this), this.trigger()));
  }
  /**
   * @internal
   */
  notify() {
    this.flags & 2 && !(this.flags & 32) || this.flags & 8 || $h(this);
  }
  run() {
    if (!(this.flags & 1))
      return this.fn();
    this.flags |= 2, jc(this), Zh(this);
    const e = ft, t = Sn;
    ft = this, Sn = !0;
    try {
      return this.fn();
    } finally {
      Jh(this), ft = e, Sn = t, this.flags &= -3;
    }
  }
  stop() {
    if (this.flags & 1) {
      for (let e = this.deps; e; e = e.nextDep)
        lc(e);
      this.deps = this.depsTail = void 0, jc(this), this.onStop && this.onStop(), this.flags &= -2;
    }
  }
  trigger() {
    this.flags & 64 ? ca.add(this) : this.scheduler ? this.scheduler() : this.runIfDirty();
  }
  /**
   * @internal
   */
  runIfDirty() {
    tl(this) && this.run();
  }
  get dirty() {
    return tl(this);
  }
}
let Kh = 0, lr, cr;
function $h(n, e = !1) {
  if (n.flags |= 8, e) {
    n.next = cr, cr = n;
    return;
  }
  n.next = lr, lr = n;
}
function oc() {
  Kh++;
}
function ac() {
  if (--Kh > 0)
    return;
  if (cr) {
    let e = cr;
    for (cr = void 0; e; ) {
      const t = e.next;
      e.next = void 0, e.flags &= -9, e = t;
    }
  }
  let n;
  for (; lr; ) {
    let e = lr;
    for (lr = void 0; e; ) {
      const t = e.next;
      if (e.next = void 0, e.flags &= -9, e.flags & 1)
        try {
          e.trigger();
        } catch (i) {
          n || (n = i);
        }
      e = t;
    }
  }
  if (n) throw n;
}
function Zh(n) {
  for (let e = n.deps; e; e = e.nextDep)
    e.version = -1, e.prevActiveLink = e.dep.activeLink, e.dep.activeLink = e;
}
function Jh(n) {
  let e, t = n.depsTail, i = t;
  for (; i; ) {
    const s = i.prevDep;
    i.version === -1 ? (i === t && (t = s), lc(i), lp(i)) : e = i, i.dep.activeLink = i.prevActiveLink, i.prevActiveLink = void 0, i = s;
  }
  n.deps = e, n.depsTail = t;
}
function tl(n) {
  for (let e = n.deps; e; e = e.nextDep)
    if (e.dep.version !== e.version || e.dep.computed && (Qh(e.dep.computed) || e.dep.version !== e.version))
      return !0;
  return !!n._dirty;
}
function Qh(n) {
  if (n.flags & 4 && !(n.flags & 16) || (n.flags &= -17, n.globalVersion === _r) || (n.globalVersion = _r, !n.isSSR && n.flags & 128 && (!n.deps && !n._dirty || !tl(n))))
    return;
  n.flags |= 2;
  const e = n.dep, t = ft, i = Sn;
  ft = n, Sn = !0;
  try {
    Zh(n);
    const s = n.fn(n._value);
    (e.version === 0 || In(s, n._value)) && (n.flags |= 128, n._value = s, e.version++);
  } catch (s) {
    throw e.version++, s;
  } finally {
    ft = t, Sn = i, Jh(n), n.flags &= -3;
  }
}
function lc(n, e = !1) {
  const { dep: t, prevSub: i, nextSub: s } = n;
  if (i && (i.nextSub = s, n.prevSub = void 0), s && (s.prevSub = i, n.nextSub = void 0), t.subs === n && (t.subs = i, !i && t.computed)) {
    t.computed.flags &= -5;
    for (let r = t.computed.deps; r; r = r.nextDep)
      lc(r, !0);
  }
  !e && !--t.sc && t.map && t.map.delete(t.key);
}
function lp(n) {
  const { prevDep: e, nextDep: t } = n;
  e && (e.nextDep = t, n.prevDep = void 0), t && (t.prevDep = e, n.nextDep = void 0);
}
let Sn = !0;
const ef = [];
function ni() {
  ef.push(Sn), Sn = !1;
}
function ii() {
  const n = ef.pop();
  Sn = n === void 0 ? !0 : n;
}
function jc(n) {
  const { cleanup: e } = n;
  if (n.cleanup = void 0, e) {
    const t = ft;
    ft = void 0;
    try {
      e();
    } finally {
      ft = t;
    }
  }
}
let _r = 0;
class cp {
  constructor(e, t) {
    this.sub = e, this.dep = t, this.version = t.version, this.nextDep = this.prevDep = this.nextSub = this.prevSub = this.prevActiveLink = void 0;
  }
}
class cc {
  // TODO isolatedDeclarations "__v_skip"
  constructor(e) {
    this.computed = e, this.version = 0, this.activeLink = void 0, this.subs = void 0, this.map = void 0, this.key = void 0, this.sc = 0, this.__v_skip = !0;
  }
  track(e) {
    if (!ft || !Sn || ft === this.computed)
      return;
    let t = this.activeLink;
    if (t === void 0 || t.sub !== ft)
      t = this.activeLink = new cp(ft, this), ft.deps ? (t.prevDep = ft.depsTail, ft.depsTail.nextDep = t, ft.depsTail = t) : ft.deps = ft.depsTail = t, tf(t);
    else if (t.version === -1 && (t.version = this.version, t.nextDep)) {
      const i = t.nextDep;
      i.prevDep = t.prevDep, t.prevDep && (t.prevDep.nextDep = i), t.prevDep = ft.depsTail, t.nextDep = void 0, ft.depsTail.nextDep = t, ft.depsTail = t, ft.deps === t && (ft.deps = i);
    }
    return t;
  }
  trigger(e) {
    this.version++, _r++, this.notify(e);
  }
  notify(e) {
    oc();
    try {
      for (let t = this.subs; t; t = t.prevSub)
        t.sub.notify() && t.sub.dep.notify();
    } finally {
      ac();
    }
  }
}
function tf(n) {
  if (n.dep.sc++, n.sub.flags & 4) {
    const e = n.dep.computed;
    if (e && !n.dep.subs) {
      e.flags |= 20;
      for (let i = e.deps; i; i = i.nextDep)
        tf(i);
    }
    const t = n.dep.subs;
    t !== n && (n.prevSub = t, t && (t.nextSub = n)), n.dep.subs = n;
  }
}
const nl = /* @__PURE__ */ new WeakMap(), Gi = /* @__PURE__ */ Symbol(
  ""
), il = /* @__PURE__ */ Symbol(
  ""
), gr = /* @__PURE__ */ Symbol(
  ""
);
function It(n, e, t) {
  if (Sn && ft) {
    let i = nl.get(n);
    i || nl.set(n, i = /* @__PURE__ */ new Map());
    let s = i.get(t);
    s || (i.set(t, s = new cc()), s.map = i, s.key = t), s.track();
  }
}
function Zn(n, e, t, i, s, r) {
  const o = nl.get(n);
  if (!o) {
    _r++;
    return;
  }
  const a = (l) => {
    l && l.trigger();
  };
  if (oc(), e === "clear")
    o.forEach(a);
  else {
    const l = ze(n), c = l && sc(t);
    if (l && t === "length") {
      const u = Number(i);
      o.forEach((h, f) => {
        (f === "length" || f === gr || !On(f) && f >= u) && a(h);
      });
    } else
      switch ((t !== void 0 || o.has(void 0)) && a(o.get(t)), c && a(o.get(gr)), e) {
        case "add":
          l ? c && a(o.get("length")) : (a(o.get(Gi)), Rs(n) && a(o.get(il)));
          break;
        case "delete":
          l || (a(o.get(Gi)), Rs(n) && a(o.get(il)));
          break;
        case "set":
          Rs(n) && a(o.get(Gi));
          break;
      }
  }
  ac();
}
function ts(n) {
  const e = /* @__PURE__ */ tt(n);
  return e === n ? e : (It(e, "iterate", gr), /* @__PURE__ */ pn(n) ? e : e.map(Tn));
}
function qo(n) {
  return It(n = /* @__PURE__ */ tt(n), "iterate", gr), n;
}
function Cn(n, e) {
  return /* @__PURE__ */ si(n) ? Us(/* @__PURE__ */ Wi(n) ? Tn(e) : e) : Tn(e);
}
const up = {
  __proto__: null,
  [Symbol.iterator]() {
    return ua(this, Symbol.iterator, (n) => Cn(this, n));
  },
  concat(...n) {
    return ts(this).concat(
      ...n.map((e) => ze(e) ? ts(e) : e)
    );
  },
  entries() {
    return ua(this, "entries", (n) => (n[1] = Cn(this, n[1]), n));
  },
  every(n, e) {
    return Vn(this, "every", n, e, void 0, arguments);
  },
  filter(n, e) {
    return Vn(
      this,
      "filter",
      n,
      e,
      (t) => t.map((i) => Cn(this, i)),
      arguments
    );
  },
  find(n, e) {
    return Vn(
      this,
      "find",
      n,
      e,
      (t) => Cn(this, t),
      arguments
    );
  },
  findIndex(n, e) {
    return Vn(this, "findIndex", n, e, void 0, arguments);
  },
  findLast(n, e) {
    return Vn(
      this,
      "findLast",
      n,
      e,
      (t) => Cn(this, t),
      arguments
    );
  },
  findLastIndex(n, e) {
    return Vn(this, "findLastIndex", n, e, void 0, arguments);
  },
  // flat, flatMap could benefit from ARRAY_ITERATE but are not straight-forward to implement
  forEach(n, e) {
    return Vn(this, "forEach", n, e, void 0, arguments);
  },
  includes(...n) {
    return ha(this, "includes", n);
  },
  indexOf(...n) {
    return ha(this, "indexOf", n);
  },
  join(n) {
    return ts(this).join(n);
  },
  // keys() iterator only reads `length`, no optimization required
  lastIndexOf(...n) {
    return ha(this, "lastIndexOf", n);
  },
  map(n, e) {
    return Vn(this, "map", n, e, void 0, arguments);
  },
  pop() {
    return Ws(this, "pop");
  },
  push(...n) {
    return Ws(this, "push", n);
  },
  reduce(n, ...e) {
    return Kc(this, "reduce", n, e);
  },
  reduceRight(n, ...e) {
    return Kc(this, "reduceRight", n, e);
  },
  shift() {
    return Ws(this, "shift");
  },
  // slice could use ARRAY_ITERATE but also seems to beg for range tracking
  some(n, e) {
    return Vn(this, "some", n, e, void 0, arguments);
  },
  splice(...n) {
    return Ws(this, "splice", n);
  },
  toReversed() {
    return ts(this).toReversed();
  },
  toSorted(n) {
    return ts(this).toSorted(n);
  },
  toSpliced(...n) {
    return ts(this).toSpliced(...n);
  },
  unshift(...n) {
    return Ws(this, "unshift", n);
  },
  values() {
    return ua(this, "values", (n) => Cn(this, n));
  }
};
function ua(n, e, t) {
  const i = qo(n), s = i[e]();
  return i !== n && !/* @__PURE__ */ pn(n) && (s._next = s.next, s.next = () => {
    const r = s._next();
    return r.done || (r.value = t(r.value)), r;
  }), s;
}
const hp = Array.prototype;
function Vn(n, e, t, i, s, r) {
  const o = qo(n), a = o !== n && !/* @__PURE__ */ pn(n), l = o[e];
  if (l !== hp[e]) {
    const h = l.apply(n, r);
    return a ? Tn(h) : h;
  }
  let c = t;
  o !== n && (a ? c = function(h, f) {
    return t.call(this, Cn(n, h), f, n);
  } : t.length > 2 && (c = function(h, f) {
    return t.call(this, h, f, n);
  }));
  const u = l.call(o, c, i);
  return a && s ? s(u) : u;
}
function Kc(n, e, t, i) {
  const s = qo(n), r = s !== n && !/* @__PURE__ */ pn(n);
  let o = t, a = !1;
  s !== n && (r ? (a = i.length === 0, o = function(c, u, h) {
    return a && (a = !1, c = Cn(n, c)), t.call(this, c, Cn(n, u), h, n);
  }) : t.length > 3 && (o = function(c, u, h) {
    return t.call(this, c, u, h, n);
  }));
  const l = s[e](o, ...i);
  return a ? Cn(n, l) : l;
}
function ha(n, e, t) {
  const i = /* @__PURE__ */ tt(n);
  It(i, "iterate", gr);
  const s = i[e](...t);
  return (s === -1 || s === !1) && /* @__PURE__ */ fc(t[0]) ? (t[0] = /* @__PURE__ */ tt(t[0]), i[e](...t)) : s;
}
function Ws(n, e, t = []) {
  ni(), oc();
  const i = (/* @__PURE__ */ tt(n))[e].apply(n, t);
  return ac(), ii(), i;
}
const fp = /* @__PURE__ */ nc("__proto__,__v_isRef,__isVue"), nf = new Set(
  /* @__PURE__ */ Object.getOwnPropertyNames(Symbol).filter((n) => n !== "arguments" && n !== "caller").map((n) => Symbol[n]).filter(On)
);
function dp(n) {
  On(n) || (n = String(n));
  const e = /* @__PURE__ */ tt(this);
  return It(e, "has", n), e.hasOwnProperty(n);
}
class sf {
  constructor(e = !1, t = !1) {
    this._isReadonly = e, this._isShallow = t;
  }
  get(e, t, i) {
    if (t === "__v_skip") return e.__v_skip;
    const s = this._isReadonly, r = this._isShallow;
    if (t === "__v_isReactive")
      return !s;
    if (t === "__v_isReadonly")
      return s;
    if (t === "__v_isShallow")
      return r;
    if (t === "__v_raw")
      return i === (s ? r ? Ep : lf : r ? af : of).get(e) || // receiver is not the reactive proxy, but has the same prototype
      // this means the receiver is a user proxy of the reactive proxy
      Object.getPrototypeOf(e) === Object.getPrototypeOf(i) ? e : void 0;
    const o = ze(e);
    if (!s) {
      let l;
      if (o && (l = up[t]))
        return l;
      if (t === "hasOwnProperty")
        return dp;
    }
    const a = Reflect.get(
      e,
      t,
      // if this is a proxy wrapping a ref, return methods using the raw ref
      // as receiver so that we don't have to call `toRaw` on the ref in all
      // its class methods
      /* @__PURE__ */ Ut(e) ? e : i
    );
    if ((On(t) ? nf.has(t) : fp(t)) || (s || It(e, "get", t), r))
      return a;
    if (/* @__PURE__ */ Ut(a)) {
      const l = o && sc(t) ? a : a.value;
      return s && st(l) ? /* @__PURE__ */ rl(l) : l;
    }
    return st(a) ? s ? /* @__PURE__ */ rl(a) : /* @__PURE__ */ vr(a) : a;
  }
}
class rf extends sf {
  constructor(e = !1) {
    super(!1, e);
  }
  set(e, t, i, s) {
    let r = e[t];
    const o = ze(e) && sc(t);
    if (!this._isShallow) {
      const c = /* @__PURE__ */ si(r);
      if (!/* @__PURE__ */ pn(i) && !/* @__PURE__ */ si(i) && (r = /* @__PURE__ */ tt(r), i = /* @__PURE__ */ tt(i)), !o && /* @__PURE__ */ Ut(r) && !/* @__PURE__ */ Ut(i))
        return c || (r.value = i), !0;
    }
    const a = o ? Number(t) < e.length : it(e, t), l = Reflect.set(
      e,
      t,
      i,
      /* @__PURE__ */ Ut(e) ? e : s
    );
    return e === /* @__PURE__ */ tt(s) && l && (a ? In(i, r) && Zn(e, "set", t, i) : Zn(e, "add", t, i)), l;
  }
  deleteProperty(e, t) {
    const i = it(e, t);
    e[t];
    const s = Reflect.deleteProperty(e, t);
    return s && i && Zn(e, "delete", t, void 0), s;
  }
  has(e, t) {
    const i = Reflect.has(e, t);
    return (!On(t) || !nf.has(t)) && It(e, "has", t), i;
  }
  ownKeys(e) {
    return It(
      e,
      "iterate",
      ze(e) ? "length" : Gi
    ), Reflect.ownKeys(e);
  }
}
class pp extends sf {
  constructor(e = !1) {
    super(!0, e);
  }
  set(e, t) {
    return !0;
  }
  deleteProperty(e, t) {
    return !0;
  }
}
const mp = /* @__PURE__ */ new rf(), _p = /* @__PURE__ */ new pp(), gp = /* @__PURE__ */ new rf(!0);
const sl = (n) => n, zr = (n) => Reflect.getPrototypeOf(n);
function vp(n, e, t) {
  return function(...i) {
    const s = this.__v_raw, r = /* @__PURE__ */ tt(s), o = Rs(r), a = n === "entries" || n === Symbol.iterator && o, l = n === "keys" && o, c = s[n](...i), u = t ? sl : e ? Us : Tn;
    return !e && It(
      r,
      "iterate",
      l ? il : Gi
    ), Rt(
      // inheriting all iterator properties
      Object.create(c),
      {
        // iterator protocol
        next() {
          const { value: h, done: f } = c.next();
          return f ? { value: h, done: f } : {
            value: a ? [u(h[0]), u(h[1])] : u(h),
            done: f
          };
        }
      }
    );
  };
}
function Hr(n) {
  return function(...e) {
    return n === "delete" ? !1 : n === "clear" ? void 0 : this;
  };
}
function xp(n, e) {
  const t = {
    get(s) {
      const r = this.__v_raw, o = /* @__PURE__ */ tt(r), a = /* @__PURE__ */ tt(s);
      n || (In(s, a) && It(o, "get", s), It(o, "get", a));
      const { has: l } = zr(o), c = e ? sl : n ? Us : Tn;
      if (l.call(o, s))
        return c(r.get(s));
      if (l.call(o, a))
        return c(r.get(a));
      r !== o && r.get(s);
    },
    get size() {
      const s = this.__v_raw;
      return !n && It(/* @__PURE__ */ tt(s), "iterate", Gi), s.size;
    },
    has(s) {
      const r = this.__v_raw, o = /* @__PURE__ */ tt(r), a = /* @__PURE__ */ tt(s);
      return n || (In(s, a) && It(o, "has", s), It(o, "has", a)), s === a ? r.has(s) : r.has(s) || r.has(a);
    },
    forEach(s, r) {
      const o = this, a = o.__v_raw, l = /* @__PURE__ */ tt(a), c = e ? sl : n ? Us : Tn;
      return !n && It(l, "iterate", Gi), a.forEach((u, h) => s.call(r, c(u), c(h), o));
    }
  };
  return Rt(
    t,
    n ? {
      add: Hr("add"),
      set: Hr("set"),
      delete: Hr("delete"),
      clear: Hr("clear")
    } : {
      add(s) {
        const r = /* @__PURE__ */ tt(this), o = zr(r), a = /* @__PURE__ */ tt(s), l = !e && !/* @__PURE__ */ pn(s) && !/* @__PURE__ */ si(s) ? a : s;
        return o.has.call(r, l) || In(s, l) && o.has.call(r, s) || In(a, l) && o.has.call(r, a) || (r.add(l), Zn(r, "add", l, l)), this;
      },
      set(s, r) {
        !e && !/* @__PURE__ */ pn(r) && !/* @__PURE__ */ si(r) && (r = /* @__PURE__ */ tt(r));
        const o = /* @__PURE__ */ tt(this), { has: a, get: l } = zr(o);
        let c = a.call(o, s);
        c || (s = /* @__PURE__ */ tt(s), c = a.call(o, s));
        const u = l.call(o, s);
        return o.set(s, r), c ? In(r, u) && Zn(o, "set", s, r) : Zn(o, "add", s, r), this;
      },
      delete(s) {
        const r = /* @__PURE__ */ tt(this), { has: o, get: a } = zr(r);
        let l = o.call(r, s);
        l || (s = /* @__PURE__ */ tt(s), l = o.call(r, s)), a && a.call(r, s);
        const c = r.delete(s);
        return l && Zn(r, "delete", s, void 0), c;
      },
      clear() {
        const s = /* @__PURE__ */ tt(this), r = s.size !== 0, o = s.clear();
        return r && Zn(
          s,
          "clear",
          void 0,
          void 0
        ), o;
      }
    }
  ), [
    "keys",
    "values",
    "entries",
    Symbol.iterator
  ].forEach((s) => {
    t[s] = vp(s, n, e);
  }), t;
}
function uc(n, e) {
  const t = xp(n, e);
  return (i, s, r) => s === "__v_isReactive" ? !n : s === "__v_isReadonly" ? n : s === "__v_raw" ? i : Reflect.get(
    it(t, s) && s in i ? t : i,
    s,
    r
  );
}
const Mp = {
  get: /* @__PURE__ */ uc(!1, !1)
}, Sp = {
  get: /* @__PURE__ */ uc(!1, !0)
}, yp = {
  get: /* @__PURE__ */ uc(!0, !1)
};
const of = /* @__PURE__ */ new WeakMap(), af = /* @__PURE__ */ new WeakMap(), lf = /* @__PURE__ */ new WeakMap(), Ep = /* @__PURE__ */ new WeakMap();
function Tp(n) {
  switch (n) {
    case "Object":
    case "Array":
      return 1;
    case "Map":
    case "Set":
    case "WeakMap":
    case "WeakSet":
      return 2;
    default:
      return 0;
  }
}
// @__NO_SIDE_EFFECTS__
function vr(n) {
  return /* @__PURE__ */ si(n) ? n : hc(
    n,
    !1,
    mp,
    Mp,
    of
  );
}
// @__NO_SIDE_EFFECTS__
function bp(n) {
  return hc(
    n,
    !1,
    gp,
    Sp,
    af
  );
}
// @__NO_SIDE_EFFECTS__
function rl(n) {
  return hc(
    n,
    !0,
    _p,
    yp,
    lf
  );
}
function hc(n, e, t, i, s) {
  if (!st(n) || n.__v_raw && !(e && n.__v_isReactive) || n.__v_skip || !Object.isExtensible(n))
    return n;
  const r = s.get(n);
  if (r)
    return r;
  const o = Tp(jd(n));
  if (o === 0)
    return n;
  const a = new Proxy(
    n,
    o === 2 ? i : t
  );
  return s.set(n, a), a;
}
// @__NO_SIDE_EFFECTS__
function Wi(n) {
  return /* @__PURE__ */ si(n) ? /* @__PURE__ */ Wi(n.__v_raw) : !!(n && n.__v_isReactive);
}
// @__NO_SIDE_EFFECTS__
function si(n) {
  return !!(n && n.__v_isReadonly);
}
// @__NO_SIDE_EFFECTS__
function pn(n) {
  return !!(n && n.__v_isShallow);
}
// @__NO_SIDE_EFFECTS__
function fc(n) {
  return n ? !!n.__v_raw : !1;
}
// @__NO_SIDE_EFFECTS__
function tt(n) {
  const e = n && n.__v_raw;
  return e ? /* @__PURE__ */ tt(e) : n;
}
function Ap(n) {
  return !it(n, "__v_skip") && Object.isExtensible(n) && Wh(n, "__v_skip", !0), n;
}
const Tn = (n) => st(n) ? /* @__PURE__ */ vr(n) : n, Us = (n) => st(n) ? /* @__PURE__ */ rl(n) : n;
// @__NO_SIDE_EFFECTS__
function Ut(n) {
  return n ? n.__v_isRef === !0 : !1;
}
// @__NO_SIDE_EFFECTS__
function ns(n) {
  return wp(n, !1);
}
function wp(n, e) {
  return /* @__PURE__ */ Ut(n) ? n : new Rp(n, e);
}
class Rp {
  constructor(e, t) {
    this.dep = new cc(), this.__v_isRef = !0, this.__v_isShallow = !1, this._rawValue = t ? e : /* @__PURE__ */ tt(e), this._value = t ? e : Tn(e), this.__v_isShallow = t;
  }
  get value() {
    return this.dep.track(), this._value;
  }
  set value(e) {
    const t = this._rawValue, i = this.__v_isShallow || /* @__PURE__ */ pn(e) || /* @__PURE__ */ si(e);
    e = i ? e : /* @__PURE__ */ tt(e), In(e, t) && (this._rawValue = e, this._value = i ? e : Tn(e), this.dep.trigger());
  }
}
function yt(n) {
  return /* @__PURE__ */ Ut(n) ? n.value : n;
}
const Cp = {
  get: (n, e, t) => e === "__v_raw" ? n : yt(Reflect.get(n, e, t)),
  set: (n, e, t, i) => {
    const s = n[e];
    return /* @__PURE__ */ Ut(s) && !/* @__PURE__ */ Ut(t) ? (s.value = t, !0) : Reflect.set(n, e, t, i);
  }
};
function cf(n) {
  return /* @__PURE__ */ Wi(n) ? n : new Proxy(n, Cp);
}
class Pp {
  constructor(e, t, i) {
    this.fn = e, this.setter = t, this._value = void 0, this.dep = new cc(this), this.__v_isRef = !0, this.deps = void 0, this.depsTail = void 0, this.flags = 16, this.globalVersion = _r - 1, this.next = void 0, this.effect = this, this.__v_isReadonly = !t, this.isSSR = i;
  }
  /**
   * @internal
   */
  notify() {
    if (this.flags |= 16, !(this.flags & 8) && // avoid infinite self recursion
    ft !== this)
      return $h(this, !0), !0;
  }
  get value() {
    const e = this.dep.track();
    return Qh(this), e && (e.version = this.dep.version), this._value;
  }
  set value(e) {
    this.setter && this.setter(e);
  }
}
// @__NO_SIDE_EFFECTS__
function Dp(n, e, t = !1) {
  let i, s;
  return Ye(n) ? i = n : (i = n.get, s = n.set), new Pp(i, s, t);
}
const Vr = {}, wo = /* @__PURE__ */ new WeakMap();
let Fi;
function Lp(n, e = !1, t = Fi) {
  if (t) {
    let i = wo.get(t);
    i || wo.set(t, i = []), i.push(n);
  }
}
function Ip(n, e, t = ut) {
  const { immediate: i, deep: s, once: r, scheduler: o, augmentJob: a, call: l } = t, c = (M) => s ? M : /* @__PURE__ */ pn(M) || s === !1 || s === 0 ? Jn(M, 1) : Jn(M);
  let u, h, f, p, v = !1, x = !1;
  if (/* @__PURE__ */ Ut(n) ? (h = () => n.value, v = /* @__PURE__ */ pn(n)) : /* @__PURE__ */ Wi(n) ? (h = () => c(n), v = !0) : ze(n) ? (x = !0, v = n.some((M) => /* @__PURE__ */ Wi(M) || /* @__PURE__ */ pn(M)), h = () => n.map((M) => {
    if (/* @__PURE__ */ Ut(M))
      return M.value;
    if (/* @__PURE__ */ Wi(M))
      return c(M);
    if (Ye(M))
      return l ? l(M, 2) : M();
  })) : Ye(n) ? e ? h = l ? () => l(n, 2) : n : h = () => {
    if (f) {
      ni();
      try {
        f();
      } finally {
        ii();
      }
    }
    const M = Fi;
    Fi = u;
    try {
      return l ? l(n, 3, [p]) : n(p);
    } finally {
      Fi = M;
    }
  } : h = Fn, e && s) {
    const M = h, C = s === !0 ? 1 / 0 : s;
    h = () => Jn(M(), C);
  }
  const m = ap(), d = () => {
    u.stop(), m && m.active && ic(m.effects, u);
  };
  if (r && e) {
    const M = e;
    e = (...C) => {
      const w = M(...C);
      return d(), w;
    };
  }
  let b = x ? new Array(n.length).fill(Vr) : Vr;
  const A = (M) => {
    if (!(!(u.flags & 1) || !u.dirty && !M))
      if (e) {
        const C = u.run();
        if (M || s || v || (x ? C.some((w, P) => In(w, b[P])) : In(C, b))) {
          f && f();
          const w = Fi;
          Fi = u;
          try {
            const P = [
              C,
              // pass undefined as the old value when it's changed for the first time
              b === Vr ? void 0 : x && b[0] === Vr ? [] : b,
              p
            ];
            b = C, l ? l(e, 3, P) : (
              // @ts-expect-error
              e(...P)
            );
          } finally {
            Fi = w;
          }
        }
      } else
        u.run();
  };
  return a && a(A), u = new jh(h), u.scheduler = o ? () => o(A, !1) : A, p = (M) => Lp(M, !1, u), f = u.onStop = () => {
    const M = wo.get(u);
    if (M) {
      if (l)
        l(M, 4);
      else
        for (const C of M) C();
      wo.delete(u);
    }
  }, e ? i ? A(!0) : b = u.run() : o ? o(A.bind(null, !0), !0) : u.run(), d.pause = u.pause.bind(u), d.resume = u.resume.bind(u), d.stop = d, d;
}
function Jn(n, e = 1 / 0, t) {
  if (e <= 0 || !st(n) || n.__v_skip || (t = t || /* @__PURE__ */ new Map(), (t.get(n) || 0) >= e))
    return n;
  if (t.set(n, e), e--, /* @__PURE__ */ Ut(n))
    Jn(n.value, e, t);
  else if (ze(n))
    for (let i = 0; i < n.length; i++)
      Jn(n[i], e, t);
  else if (zh(n) || Rs(n))
    n.forEach((i) => {
      Jn(i, e, t);
    });
  else if (kh(n)) {
    for (const i in n)
      Jn(n[i], e, t);
    for (const i of Object.getOwnPropertySymbols(n))
      Object.prototype.propertyIsEnumerable.call(n, i) && Jn(n[i], e, t);
  }
  return n;
}
function Dr(n, e, t, i) {
  try {
    return i ? n(...i) : n();
  } catch (s) {
    jo(s, e, t);
  }
}
function mn(n, e, t, i) {
  if (Ye(n)) {
    const s = Dr(n, e, t, i);
    return s && Hh(s) && s.catch((r) => {
      jo(r, e, t);
    }), s;
  }
  if (ze(n)) {
    const s = [];
    for (let r = 0; r < n.length; r++)
      s.push(mn(n[r], e, t, i));
    return s;
  }
}
function jo(n, e, t, i = !0) {
  const s = e ? e.vnode : null, { errorHandler: r, throwUnhandledErrorInProduction: o } = e && e.appContext.config || ut;
  if (e) {
    let a = e.parent;
    const l = e.proxy, c = `https://vuejs.org/error-reference/#runtime-${t}`;
    for (; a; ) {
      const u = a.ec;
      if (u) {
        for (let h = 0; h < u.length; h++)
          if (u[h](n, l, c) === !1)
            return;
      }
      a = a.parent;
    }
    if (r) {
      ni(), Dr(r, null, 10, [
        n,
        l,
        c
      ]), ii();
      return;
    }
  }
  Up(n, t, s, i, o);
}
function Up(n, e, t, i = !0, s = !1) {
  if (s)
    throw n;
  console.error(n);
}
const Ht = [];
let wn = -1;
const Cs = [];
let di = null, vs = 0;
const uf = /* @__PURE__ */ Promise.resolve();
let Ro = null;
function Np(n) {
  const e = Ro || uf;
  return n ? e.then(this ? n.bind(this) : n) : e;
}
function Fp(n) {
  let e = wn + 1, t = Ht.length;
  for (; e < t; ) {
    const i = e + t >>> 1, s = Ht[i], r = xr(s);
    r < n || r === n && s.flags & 2 ? e = i + 1 : t = i;
  }
  return e;
}
function dc(n) {
  if (!(n.flags & 1)) {
    const e = xr(n), t = Ht[Ht.length - 1];
    !t || // fast path when the job id is larger than the tail
    !(n.flags & 2) && e >= xr(t) ? Ht.push(n) : Ht.splice(Fp(e), 0, n), n.flags |= 1, hf();
  }
}
function hf() {
  Ro || (Ro = uf.then(df));
}
function Op(n) {
  if (!ze(n))
    di && n.id === -1 ? di.splice(vs + 1, 0, n) : n.flags & 1 || (Cs.push(n), n.flags |= 1);
  else
    for (let e = 0; e < n.length; e++)
      Cs.push(n[e]);
  hf();
}
function $c(n, e, t = wn + 1) {
  for (; t < Ht.length; t++) {
    const i = Ht[t];
    if (i && i.flags & 2) {
      if (n && i.id !== n.uid)
        continue;
      Ht.splice(t, 1), t--, i.flags & 4 && (i.flags &= -2), i(), i.flags & 4 || (i.flags &= -2);
    }
  }
}
function ff(n) {
  if (Cs.length) {
    const e = [...new Set(Cs)].sort(
      (t, i) => xr(t) - xr(i)
    );
    if (Cs.length = 0, di) {
      for (let t = 0; t < e.length; t++)
        di.push(e[t]);
      return;
    }
    for (di = e, vs = 0; vs < di.length; vs++) {
      const t = di[vs];
      t.flags & 4 && (t.flags &= -2), t.flags & 8 || t(), t.flags &= -2;
    }
    di = null, vs = 0;
  }
}
const xr = (n) => n.id == null ? n.flags & 2 ? -1 : 1 / 0 : n.id;
function df(n) {
  try {
    for (wn = 0; wn < Ht.length; wn++) {
      const e = Ht[wn];
      e && !(e.flags & 8) && (e.flags & 4 && (e.flags &= -2), Dr(
        e,
        e.i,
        e.i ? 15 : 14
      ), e.flags & 4 || (e.flags &= -2));
    }
  } finally {
    for (; wn < Ht.length; wn++) {
      const e = Ht[wn];
      e && (e.flags &= -2);
    }
    wn = -1, Ht.length = 0, ff(), Ro = null, (Ht.length || Cs.length) && df();
  }
}
let dn = null, pf = null;
function Co(n) {
  const e = dn;
  return dn = n, pf = n && n.type.__scopeId || null, e;
}
function mf(n, e = dn, t) {
  if (!e || n._n)
    return n;
  const i = (...s) => {
    i._d && Io(-1);
    const r = Co(e), o = Xi.length;
    let a;
    try {
      a = n(...s);
    } finally {
      for (let l = Xi.length; l > o; l--) Gf();
      Co(r), i._d && Io(1);
    }
    return a;
  };
  return i._n = !0, i._c = !0, i._d = !0, i;
}
function Bp(n, e) {
  if (dn === null)
    return n;
  const t = ea(dn), i = n.dirs || (n.dirs = []);
  for (let s = 0; s < e.length; s++) {
    let [r, o, a, l = ut] = e[s];
    r && (Ye(r) && (r = {
      mounted: r,
      updated: r
    }), r.deep && Jn(o), i.push({
      dir: r,
      instance: t,
      value: o,
      oldValue: void 0,
      arg: a,
      modifiers: l
    }));
  }
  return n;
}
function Ai(n, e, t, i) {
  const s = n.dirs, r = e && e.dirs;
  for (let o = 0; o < s.length; o++) {
    const a = s[o];
    r && (a.oldValue = r[o].value);
    let l = a.dir[i];
    l && (ni(), mn(l, t, 8, [
      n.el,
      a,
      n,
      e
    ]), ii());
  }
}
function zp(n, e) {
  if (Gt) {
    let t = Gt.provides;
    const i = Gt.parent && Gt.parent.provides;
    i === t && (t = Gt.provides = Object.create(i)), t[n] = e;
  }
}
function vo(n, e, t = !1) {
  const i = Yf();
  if (i || Ps) {
    let s = Ps ? Ps._context.provides : i ? i.parent == null || i.ce ? i.vnode.appContext && i.vnode.appContext.provides : i.parent.provides : void 0;
    if (s && n in s)
      return s[n];
    if (arguments.length > 1)
      return t && Ye(e) ? e.call(i && i.proxy) : e;
  }
}
const Hp = /* @__PURE__ */ Symbol.for("v-scx"), Vp = () => vo(Hp);
function xo(n, e, t) {
  return _f(n, e, t);
}
function _f(n, e, t = ut) {
  const { immediate: i, deep: s, flush: r, once: o } = t, a = Rt({}, t), l = e && i || !e && r !== "post";
  let c;
  if (Er) {
    if (r === "sync") {
      const p = Vp();
      c = p.__watcherHandles || (p.__watcherHandles = []);
    } else if (!l) {
      const p = () => {
      };
      return p.stop = Fn, p.resume = Fn, p.pause = Fn, p;
    }
  }
  const u = Gt;
  a.call = (p, v, x) => mn(p, u, v, x);
  let h = !1;
  r === "post" ? a.scheduler = (p) => {
    Kt(p, u && u.suspense);
  } : r !== "sync" && (h = !0, a.scheduler = (p, v) => {
    v ? p() : dc(p);
  }), a.augmentJob = (p) => {
    e && (p.flags |= 4), h && (p.flags |= 2, u && (p.id = u.uid, p.i = u));
  };
  const f = Ip(n, e, a);
  return Er && (c ? c.push(f) : l && f()), f;
}
function kp(n, e, t) {
  const i = this.proxy, s = xt(n) ? n.includes(".") ? gf(i, n) : () => i[n] : n.bind(i, i);
  let r;
  Ye(e) ? r = e : (r = e.handler, t = e);
  const o = Lr(this), a = _f(s, r.bind(i), t);
  return o(), a;
}
function gf(n, e) {
  const t = e.split(".");
  return () => {
    let i = n;
    for (let s = 0; s < t.length && i; s++)
      i = i[t[s]];
    return i;
  };
}
const Gp = /* @__PURE__ */ Symbol("_vte"), Ko = (n) => n.__isTeleport, hn = /* @__PURE__ */ Symbol("_leaveCb"), Xs = /* @__PURE__ */ Symbol("_enterCb");
function Wp() {
  const n = {
    isMounted: !1,
    isLeaving: !1,
    isUnmounting: !1,
    leavingVNodes: /* @__PURE__ */ new Map()
  };
  return pc(() => {
    n.isMounted = !0;
  }), mc(() => {
    n.isUnmounting = !0;
  }), n;
}
const cn = [Function, Array], vf = {
  mode: String,
  appear: Boolean,
  persisted: Boolean,
  // enter
  onBeforeEnter: cn,
  onEnter: cn,
  onAfterEnter: cn,
  onEnterCancelled: cn,
  // leave
  onBeforeLeave: cn,
  onLeave: cn,
  onAfterLeave: cn,
  onLeaveCancelled: cn,
  // appear
  onBeforeAppear: cn,
  onAppear: cn,
  onAfterAppear: cn,
  onAppearCancelled: cn
}, xf = (n) => {
  const e = n.subTree;
  return e.component ? xf(e.component) : e;
}, Xp = {
  name: "BaseTransition",
  props: vf,
  setup(n, { slots: e }) {
    const t = Yf(), i = Wp();
    return () => {
      const s = e.default && yf(e.default(), !0), r = s && s.length ? Mf(s) : (
        // Keep explicit default-slot conditionals on the same transition path
        // as regular v-if branches, which render a comment placeholder.
        t.subTree ? sr() : void 0
      );
      if (!r)
        return;
      const o = /* @__PURE__ */ tt(n), { mode: a } = o;
      if (i.isLeaving)
        return fa(r);
      const l = Po(r);
      if (!l)
        return fa(r);
      let c = ol(
        l,
        o,
        i,
        t,
        // #11061, ensure enterHooks is fresh after clone
        (h) => c = h
      );
      l.type !== kt && Mr(l, c);
      let u = t.subTree && Po(t.subTree);
      if (u && u.type !== kt && !Bi(u, l) && xf(t).type !== kt) {
        let h = ol(
          u,
          o,
          i,
          t
        );
        if (Mr(u, h), a === "out-in" && l.type !== kt)
          return i.isLeaving = !0, h.afterLeave = () => {
            i.isLeaving = !1, t.job.flags & 8 || t.update(), delete h.afterLeave, u = void 0;
          }, fa(r);
        a === "in-out" && l.type !== kt ? h.delayLeave = (f, p, v) => {
          const x = Sf(
            i,
            u
          );
          x[String(u.key)] = u, f[hn] = () => {
            p(), f[hn] = void 0, delete c.delayedLeave, u = void 0;
          }, c.delayedLeave = () => {
            v(), delete c.delayedLeave, u = void 0;
          };
        } : u = void 0;
      } else u && (u = void 0);
      return r;
    };
  }
};
function Mf(n) {
  let e = n[0];
  if (n.length > 1) {
    for (const t of n)
      if (t.type !== kt) {
        e = t;
        break;
      }
  }
  return e;
}
const Yp = Xp;
function Sf(n, e) {
  const { leavingVNodes: t } = n;
  let i = t.get(e.type);
  return i || (i = /* @__PURE__ */ Object.create(null), t.set(e.type, i)), i;
}
function ol(n, e, t, i, s) {
  const {
    appear: r,
    mode: o,
    persisted: a = !1,
    onBeforeEnter: l,
    onEnter: c,
    onAfterEnter: u,
    onEnterCancelled: h,
    onBeforeLeave: f,
    onLeave: p,
    onAfterLeave: v,
    onLeaveCancelled: x,
    onBeforeAppear: m,
    onAppear: d,
    onAfterAppear: b,
    onAppearCancelled: A
  } = e, M = String(n.key), C = Sf(t, n), w = (S, y) => {
    S && mn(
      S,
      i,
      9,
      y
    );
  }, P = (S, y) => {
    const D = y[1];
    w(S, y), ze(S) ? S.every((L) => L.length <= 1) && D() : S.length <= 1 && D();
  }, U = {
    mode: o,
    persisted: a,
    beforeEnter(S) {
      let y = l;
      if (!t.isMounted)
        if (r)
          y = m || l;
        else
          return;
      S[hn] && S[hn](
        !0
        /* cancelled */
      );
      const D = C[M];
      D && Bi(n, D) && D.el[hn] && D.el[hn](), w(y, [S]);
    },
    enter(S) {
      if (C[M] === n) return;
      let y = c, D = u, L = h;
      if (!t.isMounted)
        if (r)
          y = d || c, D = b || u, L = A || h;
        else
          return;
      let V = !1;
      S[Xs] = (ne) => {
        V || (V = !0, ne ? w(L, [S]) : w(D, [S]), U.delayedLeave && U.delayedLeave(), S[Xs] = void 0);
      };
      const Z = S[Xs].bind(null, !1);
      y ? P(y, [S, Z]) : Z();
    },
    leave(S, y) {
      const D = String(n.key);
      if (S[Xs] && S[Xs](
        !0
        /* cancelled */
      ), t.isUnmounting)
        return y();
      w(f, [S]);
      let L = !1;
      S[hn] = (Z) => {
        L || (L = !0, y(), Z ? w(x, [S]) : w(v, [S]), S[hn] = void 0, C[D] === n && delete C[D]);
      };
      const V = S[hn].bind(null, !1);
      C[D] = n, p ? P(p, [S, V]) : V();
    },
    clone(S) {
      const y = ol(
        S,
        e,
        t,
        i,
        s
      );
      return s && s(y), y;
    }
  };
  return U;
}
function fa(n) {
  if ($o(n))
    return n = Si(n), n.children = null, n;
}
function Po(n) {
  if (!$o(n))
    return Ko(n.type) && n.children ? Mf(n.children) : n;
  if (n.component)
    return n.component.subTree;
  const { shapeFlag: e, children: t } = n;
  if (t) {
    if (e & 16)
      return t[0];
    if (e & 32 && Ye(t.default))
      return t.default();
  }
}
function Mr(n, e) {
  if (n.shapeFlag & 6 && n.component) {
    n.transition = e;
    const t = n.component.subTree;
    Mr(
      Ko(t.type) && Po(t) || t,
      e
    );
  } else n.shapeFlag & 128 ? (n.ssContent.transition = e.clone(n.ssContent), n.ssFallback.transition = e.clone(n.ssFallback)) : n.transition = e;
}
function yf(n, e = !1, t) {
  let i = [], s = 0;
  for (let r = 0; r < n.length; r++) {
    let o = n[r];
    const a = t == null ? o.key : String(t) + String(o.key != null ? o.key : r);
    o.type === Vt ? (o.patchFlag & 128 && s++, i = i.concat(
      yf(o.children, e, a)
    )) : (e || o.type !== kt) && i.push(a != null ? Si(o, { key: a }) : o);
  }
  if (s > 1)
    for (let r = 0; r < i.length; r++)
      i[r].patchFlag = -2;
  return i;
}
function Ef(n) {
  n.ids = [n.ids[0] + n.ids[2]++ + "-", 0, 0];
}
function Zc(n, e) {
  let t;
  return !!((t = Object.getOwnPropertyDescriptor(n, e)) && !t.configurable);
}
const Do = /* @__PURE__ */ new WeakMap();
function ur(n, e, t, i, s = !1) {
  if (ze(n)) {
    n.forEach(
      (x, m) => ur(
        x,
        e && (ze(e) ? e[m] : e),
        t,
        i,
        s
      )
    );
    return;
  }
  if (hr(i) && !s) {
    i.shapeFlag & 512 && i.type.__asyncResolved && i.component.subTree.component && ur(n, e, t, i.component.subTree);
    return;
  }
  const r = i.shapeFlag & 4 ? ea(i.component) : i.el, o = s ? null : r, { i: a, r: l } = n, c = e && e.r, u = a.refs === ut ? a.refs = {} : a.refs, h = a.setupState, f = /* @__PURE__ */ tt(h), p = h === ut ? Bh : (x) => Zc(u, x) ? !1 : it(f, x), v = (x, m) => !(m && Zc(u, m));
  if (c != null && c !== l) {
    if (Jc(e), xt(c))
      u[c] = null, p(c) && (h[c] = null);
    else if (/* @__PURE__ */ Ut(c)) {
      const x = e;
      v(c, x.k) && (c.value = null), x.k && (u[x.k] = null);
    }
  }
  if (Ye(l))
    Dr(l, a, 12, [o, u]);
  else {
    const x = xt(l), m = /* @__PURE__ */ Ut(l);
    if (x || m) {
      const d = () => {
        if (n.f) {
          const b = x ? p(l) ? h[l] : u[l] : v() || !n.k ? l.value : u[n.k];
          if (s)
            ze(b) && ic(b, r);
          else if (ze(b))
            b.includes(r) || b.push(r);
          else if (x)
            u[l] = [r], p(l) && (h[l] = u[l]);
          else {
            const A = [r];
            v(l, n.k) && (l.value = A), n.k && (u[n.k] = A);
          }
        } else x ? (u[l] = o, p(l) && (h[l] = o)) : m && (v(l, n.k) && (l.value = o), n.k && (u[n.k] = o));
      };
      if (o) {
        const b = () => {
          d(), Do.delete(n);
        };
        b.id = -1, Do.set(n, b), Kt(b, t);
      } else
        Jc(n), d();
    }
  }
}
function Jc(n) {
  const e = Do.get(n);
  e && (e.flags |= 8, Do.delete(n));
}
Yo().requestIdleCallback;
Yo().cancelIdleCallback;
const hr = (n) => !!n.type.__asyncLoader, $o = (n) => n.type.__isKeepAlive;
function qp(n, e) {
  Tf(n, "a", e);
}
function jp(n, e) {
  Tf(n, "da", e);
}
function Tf(n, e, t = Gt) {
  const i = n.__wdc || (n.__wdc = () => {
    let s = t;
    for (; s; ) {
      if (s.isDeactivated)
        return;
      s = s.parent;
    }
    return n();
  });
  if (Zo(e, i, t), t) {
    let s = t.parent;
    for (; s && s.parent; )
      $o(s.parent.vnode) && Kp(i, e, t, s), s = s.parent;
  }
}
function Kp(n, e, t, i) {
  const s = Zo(
    e,
    n,
    i,
    !0
    /* prepend */
  );
  bf(() => {
    ic(i[e], s);
  }, t);
}
function Zo(n, e, t = Gt, i = !1) {
  if (t) {
    const s = t[n] || (t[n] = []), r = e.__weh || (e.__weh = (...o) => {
      ni();
      const a = Lr(t), l = mn(e, t, n, o);
      return a(), ii(), l;
    });
    return i ? s.unshift(r) : s.push(r), r;
  }
}
const ri = (n) => (e, t = Gt) => {
  (!Er || n === "sp") && Zo(n, (...i) => e(...i), t);
}, $p = ri("bm"), pc = ri("m"), Zp = ri(
  "bu"
), Jp = ri("u"), mc = ri(
  "bum"
), bf = ri("um"), Qp = ri(
  "sp"
), em = ri("rtg"), tm = ri("rtc");
function nm(n, e = Gt) {
  Zo("ec", n, e);
}
const im = /* @__PURE__ */ Symbol.for("v-ndc");
function Ys(n, e, t, i) {
  let s;
  const r = t, o = ze(n);
  if (o || xt(n)) {
    const a = o && /* @__PURE__ */ Wi(n);
    let l = !1, c = !1;
    a && (l = !/* @__PURE__ */ pn(n), c = /* @__PURE__ */ si(n), n = qo(n)), s = new Array(n.length);
    for (let u = 0, h = n.length; u < h; u++)
      s[u] = e(
        l ? c ? Us(Tn(n[u])) : Tn(n[u]) : n[u],
        u,
        void 0,
        r
      );
  } else if (typeof n == "number") {
    s = new Array(n);
    for (let a = 0; a < n; a++)
      s[a] = e(a + 1, a, void 0, r);
  } else if (st(n))
    if (n[Symbol.iterator])
      s = Array.from(
        n,
        (a, l) => e(a, l, void 0, r)
      );
    else {
      const a = Object.keys(n);
      s = new Array(a.length);
      for (let l = 0, c = a.length; l < c; l++) {
        const u = a[l];
        s[l] = e(n[u], u, l, r);
      }
    }
  else
    s = [];
  return s;
}
const al = (n) => n ? qf(n) ? ea(n) : al(n.parent) : null, fr = (
  // Move PURE marker to new line to workaround compiler discarding it
  // due to type annotation
  /* @__PURE__ */ Rt(/* @__PURE__ */ Object.create(null), {
    $: (n) => n,
    $el: (n) => n.vnode.el,
    $data: (n) => n.data,
    $props: (n) => n.props,
    $attrs: (n) => n.attrs,
    $slots: (n) => n.slots,
    $refs: (n) => n.refs,
    $parent: (n) => al(n.parent),
    $root: (n) => al(n.root),
    $host: (n) => n.ce,
    $emit: (n) => n.emit,
    $options: (n) => wf(n),
    $forceUpdate: (n) => n.f || (n.f = () => {
      dc(n.update);
    }),
    $nextTick: (n) => n.n || (n.n = Np.bind(n.proxy)),
    $watch: (n) => kp.bind(n)
  })
), da = (n, e) => n !== ut && !n.__isScriptSetup && it(n, e), sm = {
  get({ _: n }, e) {
    if (e === "__v_skip")
      return !0;
    const { ctx: t, setupState: i, data: s, props: r, accessCache: o, type: a, appContext: l } = n;
    if (e[0] !== "$") {
      const f = o[e];
      if (f !== void 0)
        switch (f) {
          case 1:
            return i[e];
          case 2:
            return s[e];
          case 4:
            return t[e];
          case 3:
            return r[e];
        }
      else {
        if (da(i, e))
          return o[e] = 1, i[e];
        if (s !== ut && it(s, e))
          return o[e] = 2, s[e];
        if (it(r, e))
          return o[e] = 3, r[e];
        if (t !== ut && it(t, e))
          return o[e] = 4, t[e];
        ll && (o[e] = 0);
      }
    }
    const c = fr[e];
    let u, h;
    if (c)
      return e === "$attrs" && It(n.attrs, "get", ""), c(n);
    if (
      // css module (injected by vue-loader)
      (u = a.__cssModules) && (u = u[e])
    )
      return u;
    if (t !== ut && it(t, e))
      return o[e] = 4, t[e];
    if (
      // global properties
      h = l.config.globalProperties, it(h, e)
    )
      return h[e];
  },
  set({ _: n }, e, t) {
    const { data: i, setupState: s, ctx: r } = n;
    return da(s, e) ? (s[e] = t, !0) : i !== ut && it(i, e) ? (i[e] = t, !0) : it(n.props, e) || e[0] === "$" && e.slice(1) in n ? !1 : (r[e] = t, !0);
  },
  has({
    _: { data: n, setupState: e, accessCache: t, ctx: i, appContext: s, props: r, type: o }
  }, a) {
    let l;
    return !!(t[a] || n !== ut && a[0] !== "$" && it(n, a) || da(e, a) || it(r, a) || it(i, a) || it(fr, a) || it(s.config.globalProperties, a) || (l = o.__cssModules) && l[a]);
  },
  defineProperty(n, e, t) {
    return t.get != null ? n._.accessCache[e] = 0 : it(t, "value") && this.set(n, e, t.value, null), Reflect.defineProperty(n, e, t);
  }
};
function Qc(n) {
  return ze(n) ? n.reduce(
    (e, t) => (e[t] = null, e),
    {}
  ) : n;
}
let ll = !0;
function rm(n) {
  const e = wf(n), t = n.proxy, i = n.ctx;
  ll = !1, e.beforeCreate && eu(e.beforeCreate, n, "bc");
  const {
    // state
    data: s,
    computed: r,
    methods: o,
    watch: a,
    provide: l,
    inject: c,
    // lifecycle
    created: u,
    beforeMount: h,
    mounted: f,
    beforeUpdate: p,
    updated: v,
    activated: x,
    deactivated: m,
    beforeDestroy: d,
    beforeUnmount: b,
    destroyed: A,
    unmounted: M,
    render: C,
    renderTracked: w,
    renderTriggered: P,
    errorCaptured: U,
    serverPrefetch: S,
    // public API
    expose: y,
    inheritAttrs: D,
    // assets
    components: L,
    directives: V,
    filters: Z
  } = e;
  if (c && om(c, i, null), o)
    for (const ie in o) {
      const H = o[ie];
      Ye(H) && (i[ie] = H.bind(t));
    }
  if (s) {
    const ie = s.call(t, t);
    st(ie) && (n.data = /* @__PURE__ */ vr(ie));
  }
  if (ll = !0, r)
    for (const ie in r) {
      const H = r[ie], fe = Ye(H) ? H.bind(t, t) : Ye(H.get) ? H.get.bind(t, t) : Fn, ge = !Ye(H) && Ye(H.set) ? H.set.bind(t) : Fn, ye = nn({
        get: fe,
        set: ge
      });
      Object.defineProperty(i, ie, {
        enumerable: !0,
        configurable: !0,
        get: () => ye.value,
        set: (Fe) => ye.value = Fe
      });
    }
  if (a)
    for (const ie in a)
      Af(a[ie], i, t, ie);
  if (l) {
    const ie = Ye(l) ? l.call(t) : l;
    Reflect.ownKeys(ie).forEach((H) => {
      zp(H, ie[H]);
    });
  }
  u && eu(u, n, "c");
  function J(ie, H) {
    ze(H) ? H.forEach((fe) => ie(fe.bind(t))) : H && ie(H.bind(t));
  }
  if (J($p, h), J(pc, f), J(Zp, p), J(Jp, v), J(qp, x), J(jp, m), J(nm, U), J(tm, w), J(em, P), J(mc, b), J(bf, M), J(Qp, S), ze(y))
    if (y.length) {
      const ie = n.exposed || (n.exposed = {});
      y.forEach((H) => {
        Object.defineProperty(ie, H, {
          get: () => t[H],
          set: (fe) => t[H] = fe,
          enumerable: !0
        });
      });
    } else n.exposed || (n.exposed = {});
  C && n.render === Fn && (n.render = C), D != null && (n.inheritAttrs = D), L && (n.components = L), V && (n.directives = V), S && Ef(n);
}
function om(n, e, t = Fn) {
  ze(n) && (n = cl(n));
  for (const i in n) {
    const s = n[i];
    let r;
    st(s) ? "default" in s ? r = vo(
      s.from || i,
      s.default,
      !0
    ) : r = vo(s.from || i) : r = vo(s), /* @__PURE__ */ Ut(r) ? Object.defineProperty(e, i, {
      enumerable: !0,
      configurable: !0,
      get: () => r.value,
      set: (o) => r.value = o
    }) : e[i] = r;
  }
}
function eu(n, e, t) {
  mn(
    ze(n) ? n.map((i) => i.bind(e.proxy)) : n.bind(e.proxy),
    e,
    t
  );
}
function Af(n, e, t, i) {
  let s = i.includes(".") ? gf(t, i) : () => t[i];
  if (xt(n)) {
    const r = e[n];
    Ye(r) && xo(s, r);
  } else if (Ye(n))
    xo(s, n.bind(t));
  else if (st(n))
    if (ze(n))
      n.forEach((r) => Af(r, e, t, i));
    else {
      const r = Ye(n.handler) ? n.handler.bind(t) : e[n.handler];
      Ye(r) && xo(s, r, n);
    }
}
function wf(n) {
  const e = n.type, { mixins: t, extends: i } = e, {
    mixins: s,
    optionsCache: r,
    config: { optionMergeStrategies: o }
  } = n.appContext, a = r.get(e);
  let l;
  return a ? l = a : !s.length && !t && !i ? l = e : (l = {}, s.length && s.forEach(
    (c) => Lo(l, c, o, !0)
  ), Lo(l, e, o)), st(e) && r.set(e, l), l;
}
function Lo(n, e, t, i = !1) {
  const { mixins: s, extends: r } = e;
  r && Lo(n, r, t, !0), s && s.forEach(
    (o) => Lo(n, o, t, !0)
  );
  for (const o in e)
    if (!(i && o === "expose")) {
      const a = am[o] || t && t[o];
      n[o] = a ? a(n[o], e[o]) : e[o];
    }
  return n;
}
const am = {
  data: tu,
  props: nu,
  emits: nu,
  // objects
  methods: ir,
  computed: ir,
  // lifecycle
  beforeCreate: Bt,
  created: Bt,
  beforeMount: Bt,
  mounted: Bt,
  beforeUpdate: Bt,
  updated: Bt,
  beforeDestroy: Bt,
  beforeUnmount: Bt,
  destroyed: Bt,
  unmounted: Bt,
  activated: Bt,
  deactivated: Bt,
  errorCaptured: Bt,
  serverPrefetch: Bt,
  // assets
  components: ir,
  directives: ir,
  // watch
  watch: cm,
  // provide / inject
  provide: tu,
  inject: lm
};
function tu(n, e) {
  return e ? n ? function() {
    return Rt(
      Ye(n) ? n.call(this, this) : n,
      Ye(e) ? e.call(this, this) : e
    );
  } : e : n;
}
function lm(n, e) {
  return ir(cl(n), cl(e));
}
function cl(n) {
  if (ze(n)) {
    const e = {};
    for (let t = 0; t < n.length; t++)
      e[n[t]] = n[t];
    return e;
  }
  return n;
}
function Bt(n, e) {
  return n ? [...new Set([].concat(n, e))] : e;
}
function ir(n, e) {
  return n ? Rt(/* @__PURE__ */ Object.create(null), n, e) : e;
}
function nu(n, e) {
  return n ? ze(n) && ze(e) ? [.../* @__PURE__ */ new Set([...n, ...e])] : Rt(
    /* @__PURE__ */ Object.create(null),
    Qc(n),
    Qc(e ?? {})
  ) : e;
}
function cm(n, e) {
  if (!n) return e;
  if (!e) return n;
  const t = Rt(/* @__PURE__ */ Object.create(null), n);
  for (const i in e)
    t[i] = Bt(n[i], e[i]);
  return t;
}
function Rf() {
  return {
    app: null,
    config: {
      isNativeTag: Bh,
      performance: !1,
      globalProperties: {},
      optionMergeStrategies: {},
      errorHandler: void 0,
      warnHandler: void 0,
      compilerOptions: {}
    },
    mixins: [],
    components: {},
    directives: {},
    provides: /* @__PURE__ */ Object.create(null),
    optionsCache: /* @__PURE__ */ new WeakMap(),
    propsCache: /* @__PURE__ */ new WeakMap(),
    emitsCache: /* @__PURE__ */ new WeakMap()
  };
}
let um = 0;
function hm(n, e) {
  return function(i, s = null) {
    Ye(i) || (i = Rt({}, i)), s != null && !st(s) && (s = null);
    const r = Rf(), o = /* @__PURE__ */ new WeakSet(), a = [];
    let l = !1;
    const c = r.app = {
      _uid: um++,
      _component: i,
      _props: s,
      _container: null,
      _context: r,
      _instance: null,
      version: Wm,
      get config() {
        return r.config;
      },
      set config(u) {
      },
      use(u, ...h) {
        return o.has(u) || (u && Ye(u.install) ? (o.add(u), u.install(c, ...h)) : Ye(u) && (o.add(u), u(c, ...h))), c;
      },
      mixin(u) {
        return r.mixins.includes(u) || r.mixins.push(u), c;
      },
      component(u, h) {
        return h ? (r.components[u] = h, c) : r.components[u];
      },
      directive(u, h) {
        return h ? (r.directives[u] = h, c) : r.directives[u];
      },
      mount(u, h, f) {
        if (!l) {
          const p = c._ceVNode || $t(i, s);
          return p.appContext = r, f === !0 ? f = "svg" : f === !1 && (f = void 0), n(p, u, f), l = !0, c._container = u, u.__vue_app__ = c, ea(p.component);
        }
      },
      onUnmount(u) {
        a.push(u);
      },
      unmount() {
        l && (mn(
          a,
          c._instance,
          16
        ), n(null, c._container), delete c._container.__vue_app__);
      },
      provide(u, h) {
        return r.provides[u] = h, c;
      },
      runWithContext(u) {
        const h = Ps;
        Ps = c;
        try {
          return u();
        } finally {
          Ps = h;
        }
      }
    };
    return c;
  };
}
let Ps = null;
const fm = (n, e) => e === "modelValue" || e === "model-value" ? n.modelModifiers : n[`${e}Modifiers`] || n[`${Mn(e)}Modifiers`] || n[`${$i(e)}Modifiers`];
function dm(n, e, ...t) {
  if (n.isUnmounted) return;
  const i = n.vnode.props || ut;
  let s = t;
  const r = e.startsWith("update:"), o = r && fm(i, e.slice(7));
  o && (o.trim && (s = t.map((u) => xt(u) ? u.trim() : u)), o.number && (s = t.map(Zd)));
  let a, l = i[a = oa(e)] || // also try camelCase event handler (#2249)
  i[a = oa(Mn(e))];
  !l && r && (l = i[a = oa($i(e))]), l && mn(
    l,
    n,
    6,
    s
  );
  const c = i[a + "Once"];
  if (c) {
    if (!n.emitted)
      n.emitted = {};
    else if (n.emitted[a])
      return;
    n.emitted[a] = !0, mn(
      c,
      n,
      6,
      s
    );
  }
}
const pm = /* @__PURE__ */ new WeakMap();
function Cf(n, e, t = !1) {
  const i = t ? pm : e.emitsCache, s = i.get(n);
  if (s !== void 0)
    return s;
  const r = n.emits;
  let o = {}, a = !1;
  if (!Ye(n)) {
    const l = (c) => {
      const u = Cf(c, e, !0);
      u && (a = !0, Rt(o, u));
    };
    !t && e.mixins.length && e.mixins.forEach(l), n.extends && l(n.extends), n.mixins && n.mixins.forEach(l);
  }
  return !r && !a ? (st(n) && i.set(n, null), null) : (ze(r) ? r.forEach((l) => o[l] = null) : Rt(o, r), st(n) && i.set(n, o), o);
}
function Jo(n, e) {
  return !n || !Go(e) ? !1 : (e = e.slice(2), e = e === "Once" ? e : e.replace(/Once$/, ""), it(n, e[0].toLowerCase() + e.slice(1)) || it(n, $i(e)) || it(n, e));
}
function iu(n) {
  const {
    type: e,
    vnode: t,
    proxy: i,
    withProxy: s,
    propsOptions: [r],
    slots: o,
    attrs: a,
    emit: l,
    render: c,
    renderCache: u,
    props: h,
    data: f,
    setupState: p,
    ctx: v,
    inheritAttrs: x
  } = n, m = Co(n);
  let d, b;
  try {
    if (t.shapeFlag & 4) {
      const M = s || i, C = M;
      d = Pn(
        c.call(
          C,
          M,
          u,
          h,
          p,
          f,
          v
        )
      ), b = a;
    } else {
      const M = e;
      d = Pn(
        M.length > 1 ? M(
          h,
          { attrs: a, slots: o, emit: l }
        ) : M(
          h,
          null
        )
      ), b = e.props ? a : mm(a);
    }
  } catch (M) {
    Xi.length = 0, jo(M, n, 1), d = $t(kt);
  }
  let A = d;
  if (b && x !== !1) {
    const M = Object.keys(b), { shapeFlag: C } = A;
    M.length && C & 7 && (r && M.some(Wo) && (b = _m(
      b,
      r
    )), A = Si(A, b, !1, !0));
  }
  if (t.dirs && (A = Si(A, null, !1, !0), A.dirs = A.dirs ? A.dirs.concat(t.dirs) : t.dirs), t.transition) {
    const M = Ko(A.type) && Po(A) || A;
    Mr(M, t.transition);
  }
  return d = A, Co(m), d;
}
const mm = (n) => {
  let e;
  for (const t in n)
    (t === "class" || t === "style" || Go(t)) && ((e || (e = {}))[t] = n[t]);
  return e;
}, _m = (n, e) => {
  const t = {};
  for (const i in n)
    (!Wo(i) || !(i.slice(9) in e)) && (t[i] = n[i]);
  return t;
};
function gm(n, e, t) {
  const { props: i, children: s, component: r } = n, { props: o, children: a, patchFlag: l } = e, c = r.emitsOptions;
  if (e.dirs || e.transition)
    return !0;
  if (t && l >= 0) {
    if (l & 1024)
      return !0;
    if (l & 16)
      return i ? su(i, o, c) : !!o;
    if (l & 8) {
      const u = e.dynamicProps;
      for (let h = 0; h < u.length; h++) {
        const f = u[h];
        if (Pf(o, i, f) && !Jo(c, f))
          return !0;
      }
    }
  } else
    return (s || a) && (!a || !a.$stable) ? !0 : i === o ? !1 : i ? o ? su(i, o, c) : !0 : !!o;
  return !1;
}
function su(n, e, t) {
  const i = Object.keys(e);
  if (i.length !== Object.keys(n).length)
    return !0;
  for (let s = 0; s < i.length; s++) {
    const r = i[s];
    if (Pf(e, n, r) && !Jo(t, r))
      return !0;
  }
  return !1;
}
function Pf(n, e, t) {
  const i = n[t], s = e[t];
  return t === "style" && st(i) && st(s) ? !rc(i, s) : i !== s;
}
function vm({ vnode: n, parent: e, suspense: t }, i) {
  for (; e; ) {
    const s = e.subTree;
    if (s.suspense && s.suspense.activeBranch === n && (s.suspense.vnode.el = s.el = i, n = s), s === n)
      (n = e.vnode).el = i, e = e.parent;
    else
      break;
  }
  t && t.activeBranch === n && (t.vnode.el = i);
}
const Df = {}, Lf = () => Object.create(Df), If = (n) => Object.getPrototypeOf(n) === Df;
function xm(n, e, t, i = !1) {
  const s = {}, r = Lf();
  n.propsDefaults = /* @__PURE__ */ Object.create(null), Uf(n, e, s, r);
  for (const o in n.propsOptions[0])
    o in s || (s[o] = void 0);
  t ? n.props = i ? s : /* @__PURE__ */ bp(s) : n.type.props ? n.props = s : n.props = r, n.attrs = r;
}
function Mm(n, e, t, i) {
  const {
    props: s,
    attrs: r,
    vnode: { patchFlag: o }
  } = n, a = /* @__PURE__ */ tt(s), [l] = n.propsOptions;
  let c = !1;
  if (
    // always force full diff in dev
    // - #1942 if hmr is enabled with sfc component
    // - vite#872 non-sfc component used by sfc component
    (i || o > 0) && !(o & 16)
  ) {
    if (o & 8) {
      const u = n.vnode.dynamicProps;
      for (let h = 0; h < u.length; h++) {
        let f = u[h];
        if (Jo(n.emitsOptions, f))
          continue;
        const p = e[f];
        if (l)
          if (it(r, f))
            p !== r[f] && (r[f] = p, c = !0);
          else {
            const v = Mn(f);
            s[v] = ul(
              l,
              a,
              v,
              p,
              n,
              !1
            );
          }
        else
          p !== r[f] && (r[f] = p, c = !0);
      }
    }
  } else {
    Uf(n, e, s, r) && (c = !0);
    let u;
    for (const h in a)
      (!e || // for camelCase
      !it(e, h) && // it's possible the original props was passed in as kebab-case
      // and converted to camelCase (#955)
      ((u = $i(h)) === h || !it(e, u))) && (l ? t && // for camelCase
      (t[h] !== void 0 || // for kebab-case
      t[u] !== void 0) && (s[h] = ul(
        l,
        a,
        h,
        void 0,
        n,
        !0
      )) : delete s[h]);
    if (r !== a)
      for (const h in r)
        (!e || !it(e, h)) && (delete r[h], c = !0);
  }
  c && Zn(n.attrs, "set", "");
}
function Uf(n, e, t, i) {
  const [s, r] = n.propsOptions;
  let o = !1, a;
  if (e)
    for (let l in e) {
      if (ar(l))
        continue;
      const c = e[l];
      let u;
      s && it(s, u = Mn(l)) ? !r || !r.includes(u) ? t[u] = c : (a || (a = {}))[u] = c : Jo(n.emitsOptions, l) || (!(l in i) || c !== i[l]) && (i[l] = c, o = !0);
    }
  if (r) {
    const l = /* @__PURE__ */ tt(t), c = a || ut;
    for (let u = 0; u < r.length; u++) {
      const h = r[u];
      t[h] = ul(
        s,
        l,
        h,
        c[h],
        n,
        !it(c, h)
      );
    }
  }
  return o;
}
function ul(n, e, t, i, s, r) {
  const o = n[t];
  if (o != null) {
    const a = it(o, "default");
    if (a && i === void 0) {
      const l = o.default;
      if (o.type !== Function && !o.skipFactory && Ye(l)) {
        const { propsDefaults: c } = s;
        if (t in c)
          i = c[t];
        else {
          const u = Lr(s);
          i = c[t] = l.call(
            null,
            e
          ), u();
        }
      } else
        i = l;
      s.ce && s.ce._setProp(t, i);
    }
    o[
      0
      /* shouldCast */
    ] && (r && !a ? i = !1 : o[
      1
      /* shouldCastTrue */
    ] && (i === "" || i === $i(t)) && (i = !0));
  }
  return i;
}
const Sm = /* @__PURE__ */ new WeakMap();
function Nf(n, e, t = !1) {
  const i = t ? Sm : e.propsCache, s = i.get(n);
  if (s)
    return s;
  const r = n.props, o = {}, a = [];
  let l = !1;
  if (!Ye(n)) {
    const u = (h) => {
      l = !0;
      const [f, p] = Nf(h, e, !0);
      Rt(o, f), p && a.push(...p);
    };
    !t && e.mixins.length && e.mixins.forEach(u), n.extends && u(n.extends), n.mixins && n.mixins.forEach(u);
  }
  if (!r && !l)
    return st(n) && i.set(n, ws), ws;
  if (ze(r))
    for (let u = 0; u < r.length; u++) {
      const h = Mn(r[u]);
      ru(h) && (o[h] = ut);
    }
  else if (r)
    for (const u in r) {
      const h = Mn(u);
      if (ru(h)) {
        const f = r[u], p = o[h] = ze(f) || Ye(f) ? { type: f } : Rt({}, f), v = p.type;
        let x = !1, m = !0;
        if (ze(v))
          for (let d = 0; d < v.length; ++d) {
            const b = v[d], A = Ye(b) && b.name;
            if (A === "Boolean") {
              x = !0;
              break;
            } else A === "String" && (m = !1);
          }
        else
          x = Ye(v) && v.name === "Boolean";
        p[
          0
          /* shouldCast */
        ] = x, p[
          1
          /* shouldCastTrue */
        ] = m, (x || it(p, "default")) && a.push(h);
      }
    }
  const c = [o, a];
  return st(n) && i.set(n, c), c;
}
function ru(n) {
  return n[0] !== "$" && !ar(n);
}
const _c = (n) => n === "_" || n === "_ctx" || n === "$stable", gc = (n) => ze(n) ? n.map(Pn) : [Pn(n)], ym = (n, e, t) => {
  if (e._n)
    return e;
  const i = mf((...s) => gc(e(...s)), t);
  return i._c = !1, i;
}, Ff = (n, e, t) => {
  const i = n._ctx;
  for (const s in n) {
    if (_c(s)) continue;
    const r = n[s];
    if (Ye(r))
      e[s] = ym(s, r, i);
    else if (r != null) {
      const o = gc(r);
      e[s] = () => o;
    }
  }
}, Of = (n, e) => {
  const t = gc(e);
  n.slots.default = () => t;
}, Bf = (n, e, t) => {
  for (const i in e)
    (t || !_c(i)) && (n[i] = e[i]);
}, Em = (n, e, t) => {
  const i = n.slots = Lf();
  if (n.vnode.shapeFlag & 32) {
    const s = e._;
    s ? (Bf(i, e, t), t && Wh(i, "_", s, !0)) : Ff(e, i);
  } else e && Of(n, e);
}, Tm = (n, e, t) => {
  const { vnode: i, slots: s } = n;
  let r = !0, o = ut;
  if (i.shapeFlag & 32) {
    const a = e._;
    a ? t && a === 1 ? r = !1 : Bf(s, e, t) : (r = !e.$stable, Ff(e, s)), o = e;
  } else e && (Of(n, e), o = { default: 1 });
  if (r)
    for (const a in s)
      !_c(a) && o[a] == null && delete s[a];
}, Kt = Cm;
function bm(n) {
  return Am(n);
}
function Am(n, e) {
  const t = Yo();
  t.__VUE__ = !0;
  const {
    insert: i,
    remove: s,
    patchProp: r,
    createElement: o,
    createText: a,
    createComment: l,
    setText: c,
    setElementText: u,
    parentNode: h,
    nextSibling: f,
    setScopeId: p = Fn,
    insertStaticContent: v
  } = n, x = (R, g, W, K = null, Y = null, z = null, ae = void 0, j = null, ee = !!g.dynamicChildren) => {
    if (R === g)
      return;
    R && !Bi(R, g) && (K = re(R), Fe(R, Y, z, !0), R = null), g.patchFlag === -2 && (ee = !1, g.dynamicChildren = null);
    const { type: te, ref: xe, shapeFlag: E } = g;
    switch (te) {
      case Qo:
        m(R, g, W, K);
        break;
      case kt:
        d(R, g, W, K);
        break;
      case ma:
        R == null && b(g, W, K, ae);
        break;
      case Vt:
        L(
          R,
          g,
          W,
          K,
          Y,
          z,
          ae,
          j,
          ee
        );
        break;
      default:
        E & 1 ? C(
          R,
          g,
          W,
          K,
          Y,
          z,
          ae,
          j,
          ee
        ) : E & 6 ? V(
          R,
          g,
          W,
          K,
          Y,
          z,
          ae,
          j,
          ee
        ) : (E & 64 || E & 128) && te.process(
          R,
          g,
          W,
          K,
          Y,
          z,
          ae,
          j,
          ee,
          Pe
        );
    }
    xe != null && Y ? ur(xe, R && R.ref, z, g || R, !g) : xe == null && R && R.ref != null && ur(R.ref, null, z, R, !0);
  }, m = (R, g, W, K) => {
    if (R == null)
      i(
        g.el = a(g.children),
        W,
        K
      );
    else {
      const Y = g.el = R.el;
      g.children !== R.children && c(Y, g.children);
    }
  }, d = (R, g, W, K) => {
    R == null ? i(
      g.el = l(g.children || ""),
      W,
      K
    ) : g.el = R.el;
  }, b = (R, g, W, K) => {
    [R.el, R.anchor] = v(
      R.children,
      g,
      W,
      K,
      R.el,
      R.anchor
    );
  }, A = ({ el: R, anchor: g }, W, K) => {
    let Y;
    for (; R && R !== g; )
      Y = f(R), i(R, W, K), R = Y;
    i(g, W, K);
  }, M = ({ el: R, anchor: g }) => {
    let W;
    for (; R && R !== g; )
      W = f(R), s(R), R = W;
    s(g);
  }, C = (R, g, W, K, Y, z, ae, j, ee) => {
    if (g.type === "svg" ? ae = "svg" : g.type === "math" && (ae = "mathml"), R == null)
      w(
        g,
        W,
        K,
        Y,
        z,
        ae,
        j,
        ee
      );
    else {
      const te = R.el && R.el._isVueCE ? R.el : null;
      try {
        te && te._beginPatch(), S(
          R,
          g,
          Y,
          z,
          ae,
          j,
          ee
        );
      } finally {
        te && te._endPatch();
      }
    }
  }, w = (R, g, W, K, Y, z, ae, j) => {
    let ee, te;
    const { props: xe, shapeFlag: E, transition: _, dirs: I } = R;
    if (ee = R.el = o(
      R.type,
      z,
      xe && xe.is,
      xe
    ), E & 8 ? u(ee, R.children) : E & 16 && U(
      R.children,
      ee,
      null,
      K,
      Y,
      pa(R, z),
      ae,
      j
    ), I && Ai(R, null, K, "created"), P(ee, R, R.scopeId, ae, K), xe) {
      for (const Q in xe)
        Q !== "value" && !ar(Q) && r(ee, Q, null, xe[Q], z, K);
      "value" in xe && r(ee, "value", null, xe.value, z), (te = xe.onVnodeBeforeMount) && An(te, K, R);
    }
    I && Ai(R, null, K, "beforeMount");
    const k = wm(Y, _);
    k && _.beforeEnter(ee), i(ee, g, W), ((te = xe && xe.onVnodeMounted) || k || I) && Kt(() => {
      te && An(te, K, R), k && _.enter(ee), I && Ai(R, null, K, "mounted");
    }, Y);
  }, P = (R, g, W, K, Y) => {
    if (W && p(R, W), K)
      for (let z = 0; z < K.length; z++)
        p(R, K[z]);
    if (Y) {
      let z = Y.subTree;
      if (g === z || kf(z.type) && (z.ssContent === g || z.ssFallback === g)) {
        const ae = Y.vnode;
        P(
          R,
          ae,
          ae.scopeId,
          ae.slotScopeIds,
          Y.parent
        );
      }
    }
  }, U = (R, g, W, K, Y, z, ae, j, ee = 0) => {
    for (let te = ee; te < R.length; te++) {
      const xe = R[te] = j ? $n(R[te]) : Pn(R[te]);
      x(
        null,
        xe,
        g,
        W,
        K,
        Y,
        z,
        ae,
        j
      );
    }
  }, S = (R, g, W, K, Y, z, ae) => {
    const j = g.el = R.el;
    let { patchFlag: ee, dynamicChildren: te, dirs: xe } = g;
    ee |= R.patchFlag & 16;
    const E = R.props || ut, _ = g.props || ut;
    let I;
    if (W && wi(W, !1), (I = _.onVnodeBeforeUpdate) && An(I, W, g, R), xe && Ai(g, R, W, "beforeUpdate"), W && wi(W, !0), // #6385 the old vnode may be a user-wrapped non-isomorphic block
    // Force full diff when block metadata is unstable.
    te && (!R.dynamicChildren || R.dynamicChildren.length !== te.length) && (ee = 0, ae = !1, te = null), (E.innerHTML && _.innerHTML == null || E.textContent && _.textContent == null) && u(j, ""), te ? y(
      R.dynamicChildren,
      te,
      j,
      W,
      K,
      pa(g, Y),
      z
    ) : ae || H(
      R,
      g,
      j,
      null,
      W,
      K,
      pa(g, Y),
      z,
      !1
    ), ee > 0) {
      if (ee & 16)
        D(j, E, _, W, Y);
      else if (ee & 2 && E.class !== _.class && r(j, "class", null, _.class, Y), ee & 4 && r(j, "style", E.style, _.style, Y), ee & 8) {
        const k = g.dynamicProps;
        for (let Q = 0; Q < k.length; Q++) {
          const G = k[Q], pe = E[G], oe = _[G];
          (oe !== pe || G === "value") && r(j, G, pe, oe, Y, W);
        }
      }
      ee & 1 && R.children !== g.children && u(j, g.children);
    } else !ae && te == null && D(j, E, _, W, Y);
    ((I = _.onVnodeUpdated) || xe) && Kt(() => {
      I && An(I, W, g, R), xe && Ai(g, R, W, "updated");
    }, K);
  }, y = (R, g, W, K, Y, z, ae) => {
    for (let j = 0; j < g.length; j++) {
      const ee = R[j], te = g[j], xe = (
        // oldVNode may be an errored async setup() component inside Suspense
        // which will not have a mounted element
        ee.el && // - In the case of a Fragment, we need to provide the actual parent
        // of the Fragment itself so it can move its children.
        (ee.type === Vt || // - In the case of different nodes, there is going to be a replacement
        // which also requires the correct parent container
        !Bi(ee, te) || // - In the case of a component, it could contain anything.
        ee.shapeFlag & 198) ? h(ee.el) : (
          // In other cases, the parent container is not actually used so we
          // just pass the block element here to avoid a DOM parentNode call.
          W
        )
      );
      x(
        ee,
        te,
        xe,
        null,
        K,
        Y,
        z,
        ae,
        !0
      );
    }
  }, D = (R, g, W, K, Y) => {
    if (g !== W) {
      if (g !== ut)
        for (const z in g)
          !ar(z) && !(z in W) && r(
            R,
            z,
            g[z],
            null,
            Y,
            K
          );
      for (const z in W) {
        if (ar(z)) continue;
        const ae = W[z], j = g[z];
        ae !== j && z !== "value" && r(R, z, j, ae, Y, K);
      }
      "value" in W && r(R, "value", g.value, W.value, Y);
    }
  }, L = (R, g, W, K, Y, z, ae, j, ee) => {
    const te = g.el = R ? R.el : a(""), xe = g.anchor = R ? R.anchor : a("");
    let { patchFlag: E, dynamicChildren: _, slotScopeIds: I } = g;
    I && (j = j ? j.concat(I) : I), R == null ? (i(te, W, K), i(xe, W, K), U(
      // #10007
      // such fragment like `<></>` will be compiled into
      // a fragment which doesn't have a children.
      // In this case fallback to an empty array
      g.children || [],
      W,
      xe,
      Y,
      z,
      ae,
      j,
      ee
    )) : E > 0 && E & 64 && _ && // #2715 the previous fragment could've been a BAILed one as a result
    // of renderSlot() with no valid children
    R.dynamicChildren && R.dynamicChildren.length === _.length ? (y(
      R.dynamicChildren,
      _,
      W,
      Y,
      z,
      ae,
      j
    ), // #2080 if the stable fragment has a key, it's a <template v-for> that may
    //  get moved around. Make sure all root level vnodes inherit el.
    // #2134 or if it's a component root, it may also get moved around
    // as the component is being moved.
    (g.key != null || Y && g === Y.subTree) && zf(
      R,
      g,
      !0
      /* shallow */
    )) : H(
      R,
      g,
      W,
      xe,
      Y,
      z,
      ae,
      j,
      ee
    );
  }, V = (R, g, W, K, Y, z, ae, j, ee) => {
    g.slotScopeIds = j, R == null ? g.shapeFlag & 512 ? Y.ctx.activate(
      g,
      W,
      K,
      ae,
      ee
    ) : Z(
      g,
      W,
      K,
      Y,
      z,
      ae,
      ee
    ) : ne(R, g, ee);
  }, Z = (R, g, W, K, Y, z, ae) => {
    const j = R.component = Om(
      R,
      K,
      Y
    );
    if ($o(R) && (j.ctx.renderer = Pe), Bm(j, !1, ae), j.asyncDep) {
      if (Y && Y.registerDep(j, J, ae), !R.el) {
        const ee = j.subTree = $t(kt);
        d(null, ee, g, W), R.placeholder = ee.el;
      }
    } else
      J(
        j,
        R,
        g,
        W,
        Y,
        z,
        ae
      );
  }, ne = (R, g, W) => {
    const K = g.component = R.component;
    if (gm(R, g, W))
      if (K.asyncDep && !K.asyncResolved) {
        ie(K, g, W);
        return;
      } else
        K.next = g, K.update();
    else
      g.el = R.el, K.vnode = g;
  }, J = (R, g, W, K, Y, z, ae) => {
    const j = () => {
      if (R.isMounted) {
        let { next: E, bu: _, u: I, parent: k, vnode: Q } = R;
        {
          const Ee = Hf(R);
          if (Ee) {
            E && (E.el = Q.el, ie(R, E, ae)), Ee.asyncDep.then(() => {
              Kt(() => {
                R.isUnmounted || te();
              }, Y);
            });
            return;
          }
        }
        let G = E, pe;
        wi(R, !1), E ? (E.el = Q.el, ie(R, E, ae)) : E = Q, _ && aa(_), (pe = E.props && E.props.onVnodeBeforeUpdate) && An(pe, k, E, Q), wi(R, !0);
        const oe = iu(R), Se = R.subTree;
        R.subTree = oe, x(
          Se,
          oe,
          // parent may have changed if it's in a teleport
          h(Se.el),
          // anchor may have changed if it's in a fragment
          re(Se),
          R,
          Y,
          z
        ), E.el = oe.el, G === null && vm(R, oe.el), I && Kt(I, Y), (pe = E.props && E.props.onVnodeUpdated) && Kt(
          () => An(pe, k, E, Q),
          Y
        );
      } else {
        let E;
        const { el: _, props: I } = g, { bm: k, m: Q, parent: G, root: pe, type: oe } = R, Se = hr(g);
        wi(R, !1), k && aa(k), !Se && (E = I && I.onVnodeBeforeMount) && An(E, G, g), wi(R, !0);
        {
          pe.ce && pe.ce._hasShadowRoot() && pe.ce._injectChildStyle(
            oe,
            R.parent ? R.parent.type : void 0
          );
          const Ee = R.subTree = iu(R);
          x(
            null,
            Ee,
            W,
            K,
            R,
            Y,
            z
          ), g.el = Ee.el;
        }
        if (Q && Kt(Q, Y), !Se && (E = I && I.onVnodeMounted)) {
          const Ee = g;
          Kt(
            () => An(E, G, Ee),
            Y
          );
        }
        (g.shapeFlag & 256 || G && hr(G.vnode) && G.vnode.shapeFlag & 256) && R.a && Kt(R.a, Y), R.isMounted = !0, g = W = K = null;
      }
    };
    R.scope.on();
    const ee = R.effect = new jh(j);
    R.scope.off();
    const te = R.update = ee.run.bind(ee), xe = R.job = ee.runIfDirty.bind(ee);
    xe.i = R, xe.id = R.uid, ee.scheduler = () => dc(xe), wi(R, !0), te();
  }, ie = (R, g, W) => {
    g.component = R;
    const K = R.vnode.props;
    R.vnode = g, R.next = null, Mm(R, g.props, K, W), Tm(R, g.children, W), ni(), $c(R), ii();
  }, H = (R, g, W, K, Y, z, ae, j, ee = !1) => {
    const te = R && R.children, xe = R ? R.shapeFlag : 0, E = g.children, { patchFlag: _, shapeFlag: I } = g;
    if (_ > 0) {
      if (_ & 128) {
        ge(
          te,
          E,
          W,
          K,
          Y,
          z,
          ae,
          j,
          ee
        );
        return;
      } else if (_ & 256) {
        fe(
          te,
          E,
          W,
          K,
          Y,
          z,
          ae,
          j,
          ee
        );
        return;
      }
    }
    I & 8 ? (xe & 16 && X(te, Y, z), E !== te && u(W, E)) : xe & 16 ? I & 16 ? ge(
      te,
      E,
      W,
      K,
      Y,
      z,
      ae,
      j,
      ee
    ) : X(te, Y, z, !0) : (xe & 8 && u(W, ""), I & 16 && U(
      E,
      W,
      K,
      Y,
      z,
      ae,
      j,
      ee
    ));
  }, fe = (R, g, W, K, Y, z, ae, j, ee) => {
    R = R || ws, g = g || ws;
    const te = R.length, xe = g.length, E = Math.min(te, xe);
    let _;
    for (_ = 0; _ < E; _++) {
      const I = g[_] = ee ? $n(g[_]) : Pn(g[_]);
      x(
        R[_],
        I,
        W,
        null,
        Y,
        z,
        ae,
        j,
        ee
      );
    }
    te > xe ? X(
      R,
      Y,
      z,
      !0,
      !1,
      E
    ) : U(
      g,
      W,
      K,
      Y,
      z,
      ae,
      j,
      ee,
      E
    );
  }, ge = (R, g, W, K, Y, z, ae, j, ee) => {
    let te = 0;
    const xe = g.length;
    let E = R.length - 1, _ = xe - 1;
    for (; te <= E && te <= _; ) {
      const I = R[te], k = g[te] = ee ? $n(g[te]) : Pn(g[te]);
      if (Bi(I, k))
        x(
          I,
          k,
          W,
          null,
          Y,
          z,
          ae,
          j,
          ee
        );
      else
        break;
      te++;
    }
    for (; te <= E && te <= _; ) {
      const I = R[E], k = g[_] = ee ? $n(g[_]) : Pn(g[_]);
      if (Bi(I, k))
        x(
          I,
          k,
          W,
          null,
          Y,
          z,
          ae,
          j,
          ee
        );
      else
        break;
      E--, _--;
    }
    if (te > E) {
      if (te <= _) {
        const I = _ + 1, k = I < xe ? g[I].el : K;
        for (; te <= _; )
          x(
            null,
            g[te] = ee ? $n(g[te]) : Pn(g[te]),
            W,
            k,
            Y,
            z,
            ae,
            j,
            ee
          ), te++;
      }
    } else if (te > _)
      for (; te <= E; )
        Fe(R[te], Y, z, !0), te++;
    else {
      const I = te, k = te, Q = /* @__PURE__ */ new Map();
      for (te = k; te <= _; te++) {
        const Ce = g[te] = ee ? $n(g[te]) : Pn(g[te]);
        Ce.key != null && Q.set(Ce.key, te);
      }
      let G, pe = 0;
      const oe = _ - k + 1;
      let Se = !1, Ee = 0;
      const le = new Array(oe);
      for (te = 0; te < oe; te++) le[te] = 0;
      for (te = I; te <= E; te++) {
        const Ce = R[te];
        if (pe >= oe) {
          Fe(Ce, Y, z, !0);
          continue;
        }
        let Te;
        if (Ce.key != null)
          Te = Q.get(Ce.key);
        else
          for (G = k; G <= _; G++)
            if (le[G - k] === 0 && Bi(Ce, g[G])) {
              Te = G;
              break;
            }
        Te === void 0 ? Fe(Ce, Y, z, !0) : (le[Te - k] = te + 1, Te >= Ee ? Ee = Te : Se = !0, x(
          Ce,
          g[Te],
          W,
          null,
          Y,
          z,
          ae,
          j,
          ee
        ), pe++);
      }
      const ve = Se ? Rm(le) : ws;
      for (G = ve.length - 1, te = oe - 1; te >= 0; te--) {
        const Ce = k + te, Te = g[Ce], me = g[Ce + 1], ke = Ce + 1 < xe ? (
          // #13559, #14173 fallback to el placeholder for unresolved async component
          me.el || Vf(me)
        ) : K;
        le[te] === 0 ? x(
          null,
          Te,
          W,
          ke,
          Y,
          z,
          ae,
          j,
          ee
        ) : Se && (G < 0 || te !== ve[G] ? ye(Te, W, ke, 2) : G--);
      }
    }
  }, ye = (R, g, W, K, Y = null) => {
    const { el: z, type: ae, transition: j, children: ee, shapeFlag: te } = R;
    if (te & 6) {
      ye(R.component.subTree, g, W, K);
      return;
    }
    if (te & 128) {
      R.suspense.move(g, W, K);
      return;
    }
    if (te & 64) {
      ae.move(R, g, W, Pe);
      return;
    }
    if (ae === Vt) {
      i(z, g, W);
      for (let E = 0; E < ee.length; E++)
        ye(ee[E], g, W, K);
      i(R.anchor, g, W);
      return;
    }
    if (ae === ma) {
      A(R, g, W);
      return;
    }
    if (K !== 2 && te & 1 && j)
      if (K === 0)
        j.persisted && !z[hn] ? i(z, g, W) : (j.beforeEnter(z), i(z, g, W), Kt(() => j.enter(z), Y));
      else {
        const { leave: E, delayLeave: _, afterLeave: I } = j, k = () => {
          R.ctx.isUnmounted ? s(z) : i(z, g, W);
        }, Q = () => {
          const G = z._isLeaving || !!z[hn];
          z._isLeaving && z[hn](
            !0
            /* cancelled */
          ), j.persisted && !G ? k() : E(z, () => {
            k(), I && I();
          });
        };
        _ ? _(z, k, Q) : Q();
      }
    else
      i(z, g, W);
  }, Fe = (R, g, W, K = !1, Y = !1) => {
    const {
      type: z,
      props: ae,
      ref: j,
      children: ee,
      dynamicChildren: te,
      shapeFlag: xe,
      patchFlag: E,
      dirs: _,
      cacheIndex: I,
      memo: k
    } = R;
    if (E === -2 && (Y = !1), j != null && (ni(), ur(j, null, W, R, !0), ii()), I != null && (g.renderCache[I] = void 0), xe & 256) {
      g.ctx.deactivate(R);
      return;
    }
    const Q = xe & 1 && _, G = !hr(R);
    let pe;
    if (G && (pe = ae && ae.onVnodeBeforeUnmount) && An(pe, g, R), xe & 6)
      Ae(R.component, W, K);
    else {
      if (xe & 128) {
        R.suspense.unmount(W, K);
        return;
      }
      Q && Ai(R, null, g, "beforeUnmount"), xe & 64 ? R.type.remove(
        R,
        g,
        W,
        Pe,
        K
      ) : te && // #5154
      // when v-once is used inside a block, setBlockTracking(-1) marks the
      // parent block with hasOnce: true
      // so that it doesn't take the fast path during unmount - otherwise
      // components nested in v-once are never unmounted.
      !te.hasOnce && // #1153: fast path should not be taken for non-stable (v-for) fragments
      (z !== Vt || E > 0 && E & 64) ? X(
        te,
        g,
        W,
        !1,
        !0
      ) : (z === Vt && E & 384 || !Y && xe & 16) && X(ee, g, W), K && Je(R);
    }
    const oe = k != null && I == null;
    (G && (pe = ae && ae.onVnodeUnmounted) || Q || oe) && Kt(() => {
      pe && An(pe, g, R), Q && Ai(R, null, g, "unmounted"), oe && (R.el = null);
    }, W);
  }, Je = (R) => {
    const { type: g, el: W, anchor: K, transition: Y } = R;
    if (g === Vt) {
      Ge(W, K);
      return;
    }
    if (g === ma) {
      M(R);
      return;
    }
    const z = () => {
      s(W), Y && !Y.persisted && Y.afterLeave && Y.afterLeave();
    };
    if (R.shapeFlag & 1 && Y && !Y.persisted) {
      const { leave: ae, delayLeave: j } = Y, ee = () => ae(W, z);
      j ? j(R.el, z, ee) : ee();
    } else
      z();
  }, Ge = (R, g) => {
    let W;
    for (; R !== g; )
      W = f(R), s(R), R = W;
    s(g);
  }, Ae = (R, g, W) => {
    const { bum: K, scope: Y, job: z, subTree: ae, um: j, m: ee, a: te } = R;
    ou(ee), ou(te), K && aa(K), Y.stop(), z && (z.flags |= 8, Fe(ae, R, g, W)), j && Kt(j, g), Kt(() => {
      R.isUnmounted = !0;
    }, g);
  }, X = (R, g, W, K = !1, Y = !1, z = 0) => {
    for (let ae = z; ae < R.length; ae++)
      Fe(R[ae], g, W, K, Y);
  }, re = (R) => {
    if (R.shapeFlag & 6)
      return re(R.component.subTree);
    if (R.shapeFlag & 128)
      return R.suspense.next();
    const g = f(R.anchor || R.el), W = g && g[Gp];
    return W ? f(W) : g;
  };
  let be = !1;
  const Be = (R, g, W) => {
    let K;
    R == null ? g._vnode && (Fe(g._vnode, null, null, !0), K = g._vnode.component) : x(
      g._vnode || null,
      R,
      g,
      null,
      null,
      null,
      W
    ), g._vnode = R, be || (be = !0, $c(K), ff(), be = !1);
  }, Pe = {
    p: x,
    um: Fe,
    m: ye,
    r: Je,
    mt: Z,
    mc: U,
    pc: H,
    pbc: y,
    n: re,
    o: n
  };
  return {
    render: Be,
    hydrate: void 0,
    createApp: hm(Be)
  };
}
function pa({ type: n, props: e }, t) {
  return t === "svg" && n === "foreignObject" || t === "mathml" && n === "annotation-xml" && e && e.encoding && e.encoding.includes("html") ? void 0 : t;
}
function wi({ effect: n, job: e }, t) {
  t ? (n.flags |= 32, e.flags |= 4) : (n.flags &= -33, e.flags &= -5);
}
function wm(n, e) {
  return (!n || n && !n.pendingBranch) && e && !e.persisted;
}
function zf(n, e, t = !1) {
  const i = n.children, s = e.children;
  if (ze(i) && ze(s))
    for (let r = 0; r < i.length; r++) {
      const o = i[r];
      let a = s[r];
      a.shapeFlag & 1 && !a.dynamicChildren && ((a.patchFlag <= 0 || a.patchFlag === 32) && (a = s[r] = $n(s[r]), a.el = o.el), !t && a.patchFlag !== -2 && zf(o, a)), a.type === Qo && (a.patchFlag === -1 && (a = s[r] = $n(a)), a.el = o.el), a.type === kt && !a.el && (a.el = o.el);
    }
}
function Rm(n) {
  const e = n.slice(), t = [0];
  let i, s, r, o, a;
  const l = n.length;
  for (i = 0; i < l; i++) {
    const c = n[i];
    if (c !== 0) {
      if (s = t[t.length - 1], n[s] < c) {
        e[i] = s, t.push(i);
        continue;
      }
      for (r = 0, o = t.length - 1; r < o; )
        a = r + o >> 1, n[t[a]] < c ? r = a + 1 : o = a;
      c < n[t[r]] && (r > 0 && (e[i] = t[r - 1]), t[r] = i);
    }
  }
  for (r = t.length, o = t[r - 1]; r-- > 0; )
    t[r] = o, o = e[o];
  return t;
}
function Hf(n) {
  const e = n.subTree.component;
  if (e)
    return e.asyncDep && !e.asyncResolved ? e : Hf(e);
}
function ou(n) {
  if (n)
    for (let e = 0; e < n.length; e++)
      n[e].flags |= 8;
}
function Vf(n) {
  if (n.placeholder)
    return n.placeholder;
  const e = n.component;
  return e ? Vf(e.subTree) : null;
}
const kf = (n) => n.__isSuspense;
function Cm(n, e) {
  e && e.pendingBranch ? ze(n) ? e.effects.push(...n) : e.effects.push(n) : Op(n);
}
const Vt = /* @__PURE__ */ Symbol.for("v-fgt"), Qo = /* @__PURE__ */ Symbol.for("v-txt"), kt = /* @__PURE__ */ Symbol.for("v-cmt"), ma = /* @__PURE__ */ Symbol.for("v-stc"), Xi = [];
let on = null;
function Lt(n = !1) {
  Xi.push(on = n ? null : []);
}
function Gf() {
  Xi.pop(), on = Xi[Xi.length - 1] || null;
}
let Sr = 1;
function Io(n, e = !1) {
  Sr += n, n < 0 && on && e && (on.hasOnce = !0);
}
function Wf(n) {
  return n.dynamicChildren = Sr > 0 ? on || ws : null, Gf(), Sr > 0 && on && on.push(n), n;
}
function Ot(n, e, t, i, s, r) {
  return Wf(
    Ne(
      n,
      e,
      t,
      i,
      s,
      r,
      !0
    )
  );
}
function Pm(n, e, t, i, s) {
  return Wf(
    $t(
      n,
      e,
      t,
      i,
      s,
      !0
    )
  );
}
function Uo(n) {
  return n ? n.__v_isVNode === !0 : !1;
}
function Bi(n, e) {
  return n.type === e.type && n.key === e.key;
}
const Xf = ({ key: n }) => n ?? null, Mo = ({
  ref: n,
  ref_key: e,
  ref_for: t
}) => (typeof n == "number" && (n = "" + n), n != null ? xt(n) || /* @__PURE__ */ Ut(n) || Ye(n) ? { i: dn, r: n, k: e, f: !!t } : n : null);
function Ne(n, e = null, t = null, i = 0, s = null, r = n === Vt ? 0 : 1, o = !1, a = !1) {
  const l = {
    __v_isVNode: !0,
    __v_skip: !0,
    type: n,
    props: e,
    key: e && Xf(e),
    ref: e && Mo(e),
    scopeId: pf,
    slotScopeIds: null,
    children: t,
    component: null,
    suspense: null,
    ssContent: null,
    ssFallback: null,
    dirs: null,
    transition: null,
    el: null,
    anchor: null,
    target: null,
    targetStart: null,
    targetAnchor: null,
    staticCount: 0,
    shapeFlag: r,
    patchFlag: i,
    dynamicProps: s,
    dynamicChildren: null,
    appContext: null,
    ctx: dn
  };
  return a ? (No(l, t), r & 128 && n.normalize(l)) : t && (l.shapeFlag |= xt(t) ? 8 : 16), Sr > 0 && // avoid a block node from tracking itself
  !o && // has current parent block
  on && // presence of a patch flag indicates this node needs patching on updates.
  // component nodes also should always be patched, because even if the
  // component doesn't need to update, it needs to persist the instance on to
  // the next vnode so that it can be properly unmounted later.
  (l.patchFlag > 0 || r & 6) && // the EVENTS flag is only for hydration and if it is the only flag, the
  // vnode should not be considered dynamic due to handler caching.
  l.patchFlag !== 32 && on.push(l), l;
}
const $t = Dm;
function Dm(n, e = null, t = null, i = 0, s = null, r = !1) {
  if ((!n || n === im) && (n = kt), Uo(n)) {
    const a = Si(
      n,
      e,
      !0
      /* mergeRef: true */
    );
    return t && No(a, t), Sr > 0 && !r && on && (a.shapeFlag & 6 ? on[on.indexOf(n)] = a : on.push(a)), a.patchFlag = -2, a;
  }
  if (km(n) && (n = n.__vccOpts), e) {
    e = Lm(e);
    let { class: a, style: l } = e;
    a && !xt(a) && (e.class = mr(a)), st(l) && (/* @__PURE__ */ fc(l) && !ze(l) && (l = Rt({}, l)), e.style = gi(l));
  }
  const o = xt(n) ? 1 : kf(n) ? 128 : Ko(n) ? 64 : st(n) ? 4 : Ye(n) ? 2 : 0;
  return Ne(
    n,
    e,
    t,
    i,
    s,
    o,
    r,
    !0
  );
}
function Lm(n) {
  return n ? /* @__PURE__ */ fc(n) || If(n) ? Rt({}, n) : n : null;
}
function Si(n, e, t = !1, i = !1) {
  const { props: s, ref: r, patchFlag: o, children: a, transition: l } = n, c = e ? Um(s || {}, e) : s, u = {
    __v_isVNode: !0,
    __v_skip: !0,
    type: n.type,
    props: c,
    key: c && Xf(c),
    ref: e && e.ref ? (
      // #2078 in the case of <component :is="vnode" ref="extra"/>
      // if the vnode itself already has a ref, cloneVNode will need to merge
      // the refs so the single vnode can be set on multiple refs
      t && r ? ze(r) ? r.concat(Mo(e)) : [r, Mo(e)] : Mo(e)
    ) : r,
    scopeId: n.scopeId,
    slotScopeIds: n.slotScopeIds,
    children: a,
    target: n.target,
    targetStart: n.targetStart,
    targetAnchor: n.targetAnchor,
    staticCount: n.staticCount,
    shapeFlag: n.shapeFlag,
    // if the vnode is cloned with extra props, we can no longer assume its
    // existing patch flag to be reliable and need to add the FULL_PROPS flag.
    // note: preserve flag for fragments since they use the flag for children
    // fast paths only.
    patchFlag: e && n.type !== Vt ? o === -1 ? 16 : o | 16 : o,
    dynamicProps: n.dynamicProps,
    dynamicChildren: n.dynamicChildren,
    appContext: n.appContext,
    dirs: n.dirs,
    transition: l,
    // These should technically only be non-null on mounted VNodes. However,
    // they *should* be copied for kept-alive vnodes. So we just always copy
    // them since them being non-null during a mount doesn't affect the logic as
    // they will simply be overwritten.
    component: n.component,
    suspense: n.suspense,
    ssContent: n.ssContent && Si(n.ssContent),
    ssFallback: n.ssFallback && Si(n.ssFallback),
    placeholder: n.placeholder,
    el: n.el,
    anchor: n.anchor,
    ctx: n.ctx,
    ce: n.ce
  };
  return l && i && Mr(
    u,
    l.clone(u)
  ), u;
}
function Im(n = " ", e = 0) {
  return $t(Qo, null, n, e);
}
function sr(n = "", e = !1) {
  return e ? (Lt(), Pm(kt, null, n)) : $t(kt, null, n);
}
function Pn(n) {
  return n == null || typeof n == "boolean" ? $t(kt) : ze(n) ? $t(
    Vt,
    null,
    // #3666, avoid reference pollution when reusing vnode
    n.slice()
  ) : Uo(n) ? $n(n) : $t(Qo, null, String(n));
}
function $n(n) {
  return n.el === null && n.patchFlag !== -1 || n.memo ? n : Si(n);
}
function No(n, e) {
  let t = 0;
  const { shapeFlag: i } = n;
  if (e == null)
    e = null;
  else if (ze(e))
    t = 16;
  else if (typeof e == "object")
    if (i & 65) {
      const s = e.default;
      s && (s._c && (s._d = !1), No(n, s()), s._c && (s._d = !0));
      return;
    } else {
      t = 32;
      const s = e._;
      !s && !If(e) ? e._ctx = dn : s === 3 && dn && (dn.slots._ === 1 ? e._ = 1 : (e._ = 2, n.patchFlag |= 1024));
    }
  else if (Ye(e)) {
    if (i & 65) {
      No(n, { default: e });
      return;
    }
    e = { default: e, _ctx: dn }, t = 32;
  } else
    e = String(e), i & 64 ? (t = 16, e = [Im(e)]) : t = 8;
  n.children = e, n.shapeFlag |= t;
}
function Um(...n) {
  const e = {};
  for (let t = 0; t < n.length; t++) {
    const i = n[t];
    for (const s in i)
      if (s === "class")
        e.class !== i.class && (e.class = mr([e.class, i.class]));
      else if (s === "style")
        e.style = gi([e.style, i.style]);
      else if (Go(s)) {
        const r = e[s], o = i[s];
        o && r !== o && !(ze(r) && r.includes(o)) ? e[s] = r ? [].concat(r, o) : o : o == null && r == null && // mergeProps({ 'onUpdate:modelValue': undefined }) should not retain
        // the model listener.
        !Wo(s) && (e[s] = o);
      } else s !== "" && (e[s] = i[s]);
  }
  return e;
}
function An(n, e, t, i = null) {
  mn(n, e, 7, [
    t,
    i
  ]);
}
const Nm = Rf();
let Fm = 0;
function Om(n, e, t) {
  const i = n.type, s = (e ? e.appContext : n.appContext) || Nm, r = {
    uid: Fm++,
    vnode: n,
    type: i,
    parent: e,
    appContext: s,
    root: null,
    // to be immediately set
    next: null,
    subTree: null,
    // will be set synchronously right after creation
    effect: null,
    update: null,
    // will be set synchronously right after creation
    job: null,
    scope: new op(
      !0
      /* detached */
    ),
    render: null,
    proxy: null,
    exposed: null,
    exposeProxy: null,
    withProxy: null,
    provides: e ? e.provides : Object.create(s.provides),
    ids: e ? e.ids : ["", 0, 0],
    accessCache: null,
    renderCache: [],
    // local resolved assets
    components: null,
    directives: null,
    // resolved props and emits options
    propsOptions: Nf(i, s),
    emitsOptions: Cf(i, s),
    // emit
    emit: null,
    // to be set immediately
    emitted: null,
    // props default value
    propsDefaults: ut,
    // inheritAttrs
    inheritAttrs: i.inheritAttrs,
    // state
    ctx: ut,
    data: ut,
    props: ut,
    attrs: ut,
    slots: ut,
    refs: ut,
    setupState: ut,
    setupContext: null,
    // suspense related
    suspense: t,
    suspenseId: t ? t.pendingId : 0,
    asyncDep: null,
    asyncResolved: !1,
    // lifecycle hooks
    // not using enums here because it results in computed properties
    isMounted: !1,
    isUnmounted: !1,
    isDeactivated: !1,
    bc: null,
    c: null,
    bm: null,
    m: null,
    bu: null,
    u: null,
    um: null,
    bum: null,
    da: null,
    a: null,
    rtg: null,
    rtc: null,
    ec: null,
    sp: null
  };
  return r.ctx = { _: r }, r.root = e ? e.root : r, r.emit = dm.bind(null, r), n.ce && n.ce(r), r;
}
let Gt = null;
const Yf = () => Gt || dn;
let Fo, yr;
{
  const n = Yo(), e = (t, i) => {
    let s;
    return (s = n[t]) || (s = n[t] = []), s.push(i), (r) => {
      s.length > 1 ? s.forEach((o) => o(r)) : s[0](r);
    };
  };
  Fo = e(
    "__VUE_INSTANCE_SETTERS__",
    (t) => Gt = t
  ), yr = e(
    "__VUE_SSR_SETTERS__",
    (t) => Er = t
  );
}
const Lr = (n) => {
  const e = Gt;
  return Fo(n), n.scope.on(), () => {
    n.scope.off(), Fo(e);
  };
}, au = () => {
  Gt && Gt.scope.off(), Fo(null);
};
function qf(n) {
  return n.vnode.shapeFlag & 4;
}
let Er = !1;
function Bm(n, e = !1, t = !1) {
  e && yr(e);
  const { props: i, children: s } = n.vnode, r = qf(n);
  xm(n, i, r, e), Em(n, s, t || e);
  const o = r ? zm(n, e) : void 0;
  return e && yr(!1), o;
}
function zm(n, e) {
  const t = n.type;
  n.accessCache = /* @__PURE__ */ Object.create(null), n.proxy = new Proxy(n.ctx, sm);
  const { setup: i } = t;
  if (i) {
    ni();
    const s = n.setupContext = i.length > 1 ? Vm(n) : null, r = Lr(n), o = Dr(
      i,
      n,
      0,
      [
        n.props,
        s
      ]
    ), a = Hh(o);
    if (ii(), r(), (a || n.sp) && !hr(n) && Ef(n), a) {
      if (o.then(au, au), e)
        return o.then((l) => {
          yr(!0);
          try {
            lu(n, l, e);
          } finally {
            yr(!1);
          }
        }).catch((l) => {
          jo(l, n, 0);
        });
      n.asyncDep = o;
    } else
      lu(n, o);
  } else
    jf(n);
}
function lu(n, e, t) {
  Ye(e) ? n.type.__ssrInlineRender ? n.ssrRender = e : n.render = e : st(e) && (n.setupState = cf(e)), jf(n);
}
function jf(n, e, t) {
  const i = n.type;
  n.render || (n.render = i.render || Fn);
  {
    const s = Lr(n);
    ni();
    try {
      rm(n);
    } finally {
      ii(), s();
    }
  }
}
const Hm = {
  get(n, e) {
    return It(n, "get", ""), n[e];
  }
};
function Vm(n) {
  const e = (t) => {
    n.exposed = t || {};
  };
  return {
    attrs: new Proxy(n.attrs, Hm),
    slots: n.slots,
    emit: n.emit,
    expose: e
  };
}
function ea(n) {
  return n.exposed ? n.exposeProxy || (n.exposeProxy = new Proxy(cf(Ap(n.exposed)), {
    get(e, t) {
      if (t in e)
        return e[t];
      if (t in fr)
        return fr[t](n);
    },
    has(e, t) {
      return t in e || t in fr;
    }
  })) : n.proxy;
}
function km(n) {
  return Ye(n) && "__vccOpts" in n;
}
const nn = (n, e) => /* @__PURE__ */ Dp(n, e, Er);
function Gm(n, e, t) {
  try {
    Io(-1);
    const i = arguments.length;
    return i === 2 ? st(e) && !ze(e) ? Uo(e) ? $t(n, null, [e]) : $t(n, e) : $t(n, null, e) : (i > 3 ? t = Array.prototype.slice.call(arguments, 2) : i === 3 && Uo(t) && (t = [t]), $t(n, e, t));
  } finally {
    Io(1);
  }
}
const Wm = "3.5.41";
let hl;
const cu = typeof window < "u" && window.trustedTypes;
if (cu)
  try {
    hl = /* @__PURE__ */ cu.createPolicy("vue", {
      createHTML: (n) => n
    });
  } catch {
  }
const Kf = hl ? (n) => hl.createHTML(n) : (n) => n, Xm = "http://www.w3.org/2000/svg", Ym = "http://www.w3.org/1998/Math/MathML", Kn = typeof document < "u" ? document : null, uu = Kn && /* @__PURE__ */ Kn.createElement("template"), qm = {
  insert: (n, e, t) => {
    e.insertBefore(n, t || null);
  },
  remove: (n) => {
    const e = n.parentNode;
    e && e.removeChild(n);
  },
  createElement: (n, e, t, i) => {
    const s = e === "svg" ? Kn.createElementNS(Xm, n) : e === "mathml" ? Kn.createElementNS(Ym, n) : t ? Kn.createElement(n, { is: t }) : Kn.createElement(n);
    return n === "select" && i && i.multiple != null && s.setAttribute("multiple", i.multiple), s;
  },
  createText: (n) => Kn.createTextNode(n),
  createComment: (n) => Kn.createComment(n),
  setText: (n, e) => {
    n.nodeValue = e;
  },
  setElementText: (n, e) => {
    n.textContent = e;
  },
  parentNode: (n) => n.parentNode,
  nextSibling: (n) => n.nextSibling,
  querySelector: (n) => Kn.querySelector(n),
  setScopeId(n, e) {
    n.setAttribute(e, "");
  },
  // __UNSAFE__
  // Reason: innerHTML.
  // Static content here can only come from compiled templates.
  // As long as the user only uses trusted templates, this is safe.
  insertStaticContent(n, e, t, i, s, r) {
    const o = t ? t.previousSibling : e.lastChild;
    if (s && (s === r || s.nextSibling))
      for (; e.insertBefore(s.cloneNode(!0), t), !(s === r || !(s = s.nextSibling)); )
        ;
    else {
      uu.innerHTML = Kf(
        i === "svg" ? `<svg>${n}</svg>` : i === "mathml" ? `<math>${n}</math>` : n
      );
      const a = uu.content;
      if (i === "svg" || i === "mathml") {
        const l = a.firstChild;
        for (; l.firstChild; )
          a.appendChild(l.firstChild);
        a.removeChild(l);
      }
      e.insertBefore(a, t);
    }
    return [
      // first
      o ? o.nextSibling : e.firstChild,
      // last
      t ? t.previousSibling : e.lastChild
    ];
  }
}, oi = "transition", qs = "animation", Tr = /* @__PURE__ */ Symbol("_vtc"), $f = {
  name: String,
  type: String,
  css: {
    type: Boolean,
    default: !0
  },
  duration: [String, Number, Object],
  enterFromClass: String,
  enterActiveClass: String,
  enterToClass: String,
  appearFromClass: String,
  appearActiveClass: String,
  appearToClass: String,
  leaveFromClass: String,
  leaveActiveClass: String,
  leaveToClass: String
}, jm = /* @__PURE__ */ Rt(
  {},
  vf,
  $f
), Km = (n) => (n.displayName = "Transition", n.props = jm, n), $m = /* @__PURE__ */ Km(
  (n, { slots: e }) => Gm(Yp, Zm(n), e)
), Ri = (n, e = []) => {
  ze(n) ? n.forEach((t) => t(...e)) : n && n(...e);
}, hu = (n) => n ? ze(n) ? n.some((e) => e.length > 1) : n.length > 1 : !1;
function Zm(n) {
  const e = {};
  for (const L in n)
    L in $f || (e[L] = n[L]);
  if (n.css === !1)
    return e;
  const {
    name: t = "v",
    type: i,
    duration: s,
    enterFromClass: r = `${t}-enter-from`,
    enterActiveClass: o = `${t}-enter-active`,
    enterToClass: a = `${t}-enter-to`,
    appearFromClass: l = r,
    appearActiveClass: c = o,
    appearToClass: u = a,
    leaveFromClass: h = `${t}-leave-from`,
    leaveActiveClass: f = `${t}-leave-active`,
    leaveToClass: p = `${t}-leave-to`
  } = n, v = Jm(s), x = v && v[0], m = v && v[1], {
    onBeforeEnter: d,
    onEnter: b,
    onEnterCancelled: A,
    onLeave: M,
    onLeaveCancelled: C,
    onBeforeAppear: w = d,
    onAppear: P = b,
    onAppearCancelled: U = A
  } = e, S = (L, V, Z, ne) => {
    L._enterCancelled = ne, Ci(L, V ? u : a), Ci(L, V ? c : o), Z && Z();
  }, y = (L, V) => {
    L._isLeaving = !1, Ci(L, h), Ci(L, p), Ci(L, f), V && V();
  }, D = (L) => (V, Z) => {
    const ne = L ? P : b, J = () => S(V, L, Z);
    Ri(ne, [V, J]), fu(() => {
      Ci(V, L ? l : r), kn(V, L ? u : a), hu(ne) || du(V, i, x, J);
    });
  };
  return Rt(e, {
    onBeforeEnter(L) {
      Ri(d, [L]), kn(L, r), kn(L, o);
    },
    onBeforeAppear(L) {
      Ri(w, [L]), kn(L, l), kn(L, c);
    },
    onEnter: D(!1),
    onAppear: D(!0),
    onLeave(L, V) {
      L._isLeaving = !0;
      const Z = () => y(L, V);
      kn(L, h), L._enterCancelled ? (kn(L, f), _u(L)) : (_u(L), kn(L, f)), fu(() => {
        L._isLeaving && (Ci(L, h), kn(L, p), hu(M) || du(L, i, m, Z));
      }), Ri(M, [L, Z]);
    },
    onEnterCancelled(L) {
      S(L, !1, void 0, !0), Ri(A, [L]);
    },
    onAppearCancelled(L) {
      S(L, !0, void 0, !0), Ri(U, [L]);
    },
    onLeaveCancelled(L) {
      y(L), Ri(C, [L]);
    }
  });
}
function Jm(n) {
  if (n == null)
    return null;
  if (st(n))
    return [_a(n.enter), _a(n.leave)];
  {
    const e = _a(n);
    return [e, e];
  }
}
function _a(n) {
  return Jd(n);
}
function kn(n, e) {
  e.split(/\s+/).forEach((t) => t && n.classList.add(t)), (n[Tr] || (n[Tr] = /* @__PURE__ */ new Set())).add(e);
}
function Ci(n, e) {
  e.split(/\s+/).forEach((i) => i && n.classList.remove(i));
  const t = n[Tr];
  t && (t.delete(e), t.size || (n[Tr] = void 0));
}
function fu(n) {
  requestAnimationFrame(() => {
    requestAnimationFrame(n);
  });
}
let Qm = 0;
function du(n, e, t, i) {
  const s = n._endId = ++Qm, r = () => {
    s === n._endId && i();
  };
  if (t != null)
    return setTimeout(r, t);
  const { type: o, timeout: a, propCount: l } = e_(n, e);
  if (!o)
    return i();
  const c = o + "end";
  let u = 0;
  const h = () => {
    n.removeEventListener(c, f), r();
  }, f = (p) => {
    p.target === n && ++u >= l && h();
  };
  setTimeout(() => {
    u < l && h();
  }, a + 1), n.addEventListener(c, f);
}
function e_(n, e) {
  const t = window.getComputedStyle(n), i = (v) => (t[v] || "").split(", "), s = i(`${oi}Delay`), r = i(`${oi}Duration`), o = pu(s, r), a = i(`${qs}Delay`), l = i(`${qs}Duration`), c = pu(a, l);
  let u = null, h = 0, f = 0;
  e === oi ? o > 0 && (u = oi, h = o, f = r.length) : e === qs ? c > 0 && (u = qs, h = c, f = l.length) : (h = Math.max(o, c), u = h > 0 ? o > c ? oi : qs : null, f = u ? u === oi ? r.length : l.length : 0);
  const p = u === oi && /\b(?:transform|all)(?:,|$)/.test(
    i(`${oi}Property`).toString()
  );
  return {
    type: u,
    timeout: h,
    propCount: f,
    hasTransform: p
  };
}
function pu(n, e) {
  for (; n.length < e.length; )
    n = n.concat(n);
  return Math.max(...e.map((t, i) => mu(t) + mu(n[i])));
}
function mu(n) {
  return n === "auto" ? 0 : Number(n.slice(0, -1).replace(",", ".")) * 1e3;
}
function _u(n) {
  return (n ? n.ownerDocument : document).body.offsetHeight;
}
function t_(n, e, t) {
  const i = n[Tr];
  i && (e = (e ? [e, ...i] : [...i]).join(" ")), e == null ? n.removeAttribute("class") : t ? n.setAttribute("class", e) : n.className = e;
}
const Oo = /* @__PURE__ */ Symbol("_vod"), Zf = /* @__PURE__ */ Symbol("_vsh"), n_ = {
  // used for prop mismatch check during hydration
  name: "show",
  beforeMount(n, { value: e }, { transition: t }) {
    n[Oo] = n.style.display === "none" ? "" : n.style.display, t && e ? t.beforeEnter(n) : js(n, e);
  },
  mounted(n, { value: e }, { transition: t }) {
    t && e && t.enter(n);
  },
  updated(n, { value: e, oldValue: t }, { transition: i }) {
    !e != !t && (i ? e ? (i.beforeEnter(n), js(n, !0), i.enter(n)) : i.leave(n, () => {
      js(n, !1);
    }) : js(n, e));
  },
  beforeUnmount(n, { value: e }) {
    js(n, e);
  }
};
function js(n, e) {
  n.style.display = e ? n[Oo] : "none", n[Zf] = !e;
}
const i_ = /* @__PURE__ */ Symbol(""), s_ = /(?:^|;)\s*display\s*:/;
function r_(n, e, t) {
  const i = n.style, s = xt(t);
  let r = !1;
  if (t && !s) {
    if (e)
      if (xt(e))
        for (const o of e.split(";")) {
          const a = o.slice(0, o.indexOf(":")).trim();
          t[a] == null && rr(i, a, "");
        }
      else
        for (const o in e)
          t[o] == null && rr(i, o, "");
    for (const o in t) {
      o === "display" && (r = !0);
      const a = t[o];
      a != null ? a_(
        n,
        o,
        !xt(e) && e ? e[o] : void 0,
        a
      ) || rr(i, o, a) : rr(i, o, "");
    }
  } else if (s) {
    if (e !== t) {
      const o = i[i_];
      o && (t += ";" + o), i.cssText = t, r = s_.test(t);
    }
  } else e && n.removeAttribute("style");
  Oo in n && (n[Oo] = r ? i.display : "", n[Zf] && (i.display = "none"));
}
const gu = /\s*!important$/;
function rr(n, e, t) {
  if (ze(t))
    t.forEach((i) => rr(n, e, i));
  else if (t == null && (t = ""), e.startsWith("--"))
    n.setProperty(e, t);
  else {
    const i = o_(n, e);
    gu.test(t) ? n.setProperty(
      $i(i),
      t.replace(gu, ""),
      "important"
    ) : n[i] = t;
  }
}
const vu = ["Webkit", "Moz", "ms"], ga = {};
function o_(n, e) {
  const t = ga[e];
  if (t)
    return t;
  let i = Mn(e);
  if (i !== "filter" && i in n)
    return ga[e] = i;
  i = Gh(i);
  for (let s = 0; s < vu.length; s++) {
    const r = vu[s] + i;
    if (r in n)
      return ga[e] = r;
  }
  return e;
}
function a_(n, e, t, i) {
  return n.tagName === "TEXTAREA" && (e === "width" || e === "height") && xt(i) && t === i;
}
const xu = "http://www.w3.org/1999/xlink";
function Mu(n, e, t, i, s, r = sp(e)) {
  i && e.startsWith("xlink:") ? t == null ? n.removeAttributeNS(xu, e.slice(6, e.length)) : n.setAttributeNS(xu, e, t) : t == null || r && !Xh(t) ? n.removeAttribute(e) : n.setAttribute(
    e,
    r ? "" : On(t) ? String(t) : t
  );
}
function Su(n, e, t, i, s) {
  if (e === "innerHTML" || e === "textContent") {
    t != null && (n[e] = e === "innerHTML" ? Kf(t) : t);
    return;
  }
  const r = n.tagName;
  if (e === "value" && r !== "PROGRESS" && // custom elements may use _value internally
  !r.includes("-")) {
    const a = r === "OPTION" ? n.getAttribute("value") || "" : n.value, l = t == null ? (
      // #11647: value should be set as empty string for null and undefined,
      // but <input type="checkbox"> should be set as 'on'.
      n.type === "checkbox" ? "on" : ""
    ) : String(t);
    (a !== l || !("_value" in n)) && (n.value = l), t == null && n.removeAttribute(e), n._value = t;
    return;
  }
  let o = !1;
  if (t === "" || t == null) {
    const a = typeof n[e];
    a === "boolean" ? t = Xh(t) : t == null && a === "string" ? (t = "", o = !0) : a === "number" && (t = 0, o = !0);
  }
  try {
    n[e] = t;
  } catch {
  }
  o && n.removeAttribute(s || e);
}
function l_(n, e, t, i) {
  n.addEventListener(e, t, i);
}
function c_(n, e, t, i) {
  n.removeEventListener(e, t, i);
}
const yu = /* @__PURE__ */ Symbol("_vei");
function u_(n, e, t, i, s = null) {
  const r = n[yu] || (n[yu] = {}), o = r[e];
  if (i && o)
    o.value = i;
  else {
    const [a, l] = d_(e);
    if (i) {
      const c = r[e] = __(
        i,
        s
      );
      l_(n, a, c, l);
    } else o && (c_(n, a, o, l), r[e] = void 0);
  }
}
const h_ = /(Once|Passive|Capture)$/, f_ = /^on:?(?:Once|Passive|Capture)$/;
function d_(n) {
  let e, t;
  for (; (t = n.match(h_)) && !f_.test(n); )
    e || (e = {}), n = n.slice(0, n.length - t[1].length), e[t[1].toLowerCase()] = !0;
  return [n[2] === ":" ? n.slice(3) : $i(n.slice(2)), e];
}
let va = 0;
const p_ = /* @__PURE__ */ Promise.resolve(), m_ = () => va || (p_.then(() => va = 0), va = Date.now());
function __(n, e) {
  const t = (i) => {
    if (!i._vts)
      i._vts = Date.now();
    else if (i._vts <= t.attached)
      return;
    const s = t.value;
    if (ze(s)) {
      const r = i.stopImmediatePropagation;
      i.stopImmediatePropagation = () => {
        r.call(i), i._stopped = !0;
      };
      const o = s.slice(), a = [i];
      for (let l = 0; l < o.length && !i._stopped; l++) {
        const c = o[l];
        c && mn(
          c,
          e,
          5,
          a
        );
      }
    } else
      mn(
        s,
        e,
        5,
        [i]
      );
  };
  return t.value = n, t.attached = m_(), t;
}
const Eu = (n) => n.charCodeAt(0) === 111 && n.charCodeAt(1) === 110 && // lowercase letter
n.charCodeAt(2) > 96 && n.charCodeAt(2) < 123, g_ = (n, e, t, i, s, r) => {
  const o = s === "svg";
  e === "class" ? t_(n, i, o) : e === "style" ? r_(n, t, i) : Go(e) ? Wo(e) || u_(n, e, t, i, r) : (e[0] === "." ? (e = e.slice(1), !0) : e[0] === "^" ? (e = e.slice(1), !1) : v_(n, e, i, o)) ? (Su(n, e, i), !n.tagName.includes("-") && (e === "value" || e === "checked" || e === "selected") && Mu(n, e, i, o, r, e !== "value")) : /* #11081 force set props for possible async custom element */ n._isVueCE && // #12408 check if it's declared prop or it's async custom element
  (x_(n, e) || // @ts-expect-error _def is private
  n._def.__asyncLoader && (/[A-Z]/.test(e) || !xt(i))) ? Su(n, Mn(e), i, r, e) : (e === "true-value" ? n._trueValue = i : e === "false-value" && (n._falseValue = i), Mu(n, e, i, o));
};
function v_(n, e, t, i) {
  if (i)
    return !!(e === "innerHTML" || e === "textContent" || e in n && Eu(e) && Ye(t));
  if (e === "spellcheck" || e === "draggable" || e === "translate" || e === "autocorrect" || e === "sandbox" && n.tagName === "IFRAME" || e === "form" || e === "list" && n.tagName === "INPUT" || e === "type" && n.tagName === "TEXTAREA")
    return !1;
  if (e === "width" || e === "height") {
    const s = n.tagName;
    if (s === "IMG" || s === "VIDEO" || s === "CANVAS" || s === "SOURCE")
      return !1;
  }
  return Eu(e) && xt(t) ? !1 : e in n;
}
function x_(n, e) {
  const t = (
    // @ts-expect-error _def is private
    n._def.props
  );
  if (!t)
    return !1;
  const i = Mn(e);
  return Array.isArray(t) ? t.some((s) => Mn(s) === i) : Object.keys(t).some((s) => Mn(s) === i);
}
const M_ = ["ctrl", "shift", "alt", "meta"], S_ = {
  stop: (n) => n.stopPropagation(),
  prevent: (n) => n.preventDefault(),
  self: (n) => n.target !== n.currentTarget,
  ctrl: (n) => !n.ctrlKey,
  shift: (n) => !n.shiftKey,
  alt: (n) => !n.altKey,
  meta: (n) => !n.metaKey,
  left: (n) => "button" in n && n.button !== 0,
  middle: (n) => "button" in n && n.button !== 1,
  right: (n) => "button" in n && n.button !== 2,
  exact: (n, e) => M_.some((t) => n[`${t}Key`] && !e.includes(t))
}, y_ = (n, e) => {
  if (!n) return n;
  const t = n._withMods || (n._withMods = {}), i = e.join(".");
  return t[i] || (t[i] = ((s, ...r) => {
    for (let o = 0; o < e.length; o++) {
      const a = S_[e[o]];
      if (a && a(s, e)) return;
    }
    return n(s, ...r);
  }));
}, E_ = /* @__PURE__ */ Rt({ patchProp: g_ }, qm);
let Tu;
function T_() {
  return Tu || (Tu = bm(E_));
}
const b_ = ((...n) => {
  const e = T_().createApp(...n), { mount: t } = e;
  return e.mount = (i) => {
    const s = w_(i);
    if (!s) return;
    const r = e._component;
    !Ye(r) && !r.render && !r.template && (r.template = s.innerHTML), s.nodeType === 1 && (s.textContent = "");
    const o = t(s, !1, A_(s));
    return s instanceof Element && (s.removeAttribute("v-cloak"), s.setAttribute("data-v-app", "")), o;
  }, e;
});
function A_(n) {
  if (n instanceof SVGElement)
    return "svg";
  if (typeof MathMLElement == "function" && n instanceof MathMLElement)
    return "mathml";
}
function w_(n) {
  return xt(n) ? document.querySelector(n) : n;
}
const R_ = "AKUSPACE", C_ = "ltx25_audio", P_ = [{ id: "ltx25_audio", label: "LTX-2.5 Audio", trigger: "AKUSPACE", status: "active", supported_modes: ["dry", "room", "outside", "sfx"] }], D_ = { modes: [{ value: "dry", label: "Off" }, { value: "room", label: "Room" }, { value: "outside", label: "Space" }, { value: "sfx", label: "Sound effects" }], room_presets: ["small_room", "empty_club", "medium_room", "cathedral"], reverb_levels: ["low", "mid", "high"], outdoor_times: ["day", "night"], outdoor_level: "low", sfx_presets: ["dual_delay"], sfx_levels: ["low", "high"] }, L_ = { low: { label: "Low", caption_word: "gentle", relative_db: -25, visual_amount: 0.28 }, mid: { label: "Moderate", caption_word: "moderate", relative_db: -12, visual_amount: 0.58 }, high: { label: "Heavy", caption_word: "heavy", relative_db: 0, visual_amount: 1 } }, I_ = { dry: { mode: "dry", label: "Dry / off", short_label: "Dry", description: "Bypass the LoRA and keep the reference dry.", acoustic_fingerprint: "dry reference · no reverb", dimensions_m: [4, 5, 2.8], estimated_rt60: 0.08, estimated_predelay_ms: 1, palette: ["#d7ded9", "#202522"] }, small_room: { mode: "room", label: "Small room", short_label: "Small", description: "Bathroom-scale space with bright, close reflections.", acoustic_fingerprint: "trained caption 0.67 s · source setting ≈1.07 s", dimensions_m: [2.4, 3.2, 2.5], caption_where: "in a small bathroom-like room", caption_character: "bright close reflections and a short 0.67-second reverb decay", caption_tail: "no background ambience", estimated_rt60: 0.67, estimated_predelay_ms: 4, palette: ["#8edcff", "#173344"] }, medium_room: { mode: "room", label: "Medium room", short_label: "Medium", description: "Balanced room scale with a smooth 1.9-second decay.", acoustic_fingerprint: "trained caption 1.90 s · source setting ≈1.92 s", dimensions_m: [7, 9, 3.6], caption_where: "in a medium reverberant room", caption_character: "smooth reflections and a 1.9-second reverb decay", caption_tail: "no background ambience", estimated_rt60: 1.9, estimated_predelay_ms: 11, palette: ["#82f5c2", "#15372e"] }, empty_club: { mode: "room", label: "Empty club", short_label: "Club", description: "Filtered hard-surface reflections with a shorter, tighter 1.2-second decay.", acoustic_fingerprint: "1.20 s decay · device size 6.8", dimensions_m: [16, 24, 6], caption_where: "in an empty club", caption_character: "broad hard-surface reflections and a 1.2-second reverb decay", caption_tail: "no crowd", estimated_rt60: 1.2, estimated_predelay_ms: 18, palette: ["#ffb86f", "#422918"] }, cathedral: { mode: "room", label: "Cathedral", short_label: "Cathedral", description: "Monumental synthetic space with a long diffuse tail.", acoustic_fingerprint: "synthetic cathedral · 506 ms delay setting", dimensions_m: [26, 78, 29], caption_where: "through synthetic cathedral reverb", caption_character: "wide diffuse reflections and a long decaying tail", caption_tail: "no background ambience", estimated_rt60: 4.8, estimated_predelay_ms: 38, palette: ["#d6adff", "#392c4e"] }, dual_delay: { mode: "sfx", effect_type: "modular_dual_delay", coverage: "experimental", label: "Dual Delay", short_label: "Dual Delay", description: "Experimental modular dual-delay patch captioned during training as a modular granular delay.", acoustic_fingerprint: "modular dual delay · modular granular training caption", dimensions_m: [10, 16, 4], caption_where: "through a modular granular delay", caption_character: "scattered grains and unpredictable modulated echoes", caption_tail: "no background ambience", estimated_rt60: 0, estimated_predelay_ms: 0, palette: ["#ff78e8", "#43183e"] }, outdoor_day: { mode: "outside", time_of_day: "day", label: "Outside · day", short_label: "Day", description: "Open-air acoustics with a continuous birdsong bed.", acoustic_fingerprint: "day birds · fixed trained ambience", dimensions_m: [60, 120, 30], caption_where: "outdoors in daytime", caption_character: "open-air acoustics with continuous birdsong ambience", estimated_rt60: 0.08, estimated_predelay_ms: 2, palette: ["#ffc95c", "#3d331d"] }, outdoor_night: { mode: "outside", time_of_day: "night", label: "Outside · night", short_label: "Night", description: "Open-air acoustics with crickets and distant cars.", acoustic_fingerprint: "night crickets + cars · fixed trained ambience", dimensions_m: [60, 120, 30], caption_where: "outdoors at night", caption_character: "open-air acoustics with crickets and distant car ambience", estimated_rt60: 0.1, estimated_predelay_ms: 2, palette: ["#8f9cff", "#141a3b"] } }, Vs = {
  trigger: R_,
  default_model_profile: C_,
  model_profiles: P_,
  control_schema: D_,
  levels: L_,
  presets: I_
}, Jf = Vs.presets, dr = Vs.levels, Zi = Vs.control_schema, bu = Vs.model_profiles, U_ = Vs.default_model_profile, N_ = bu.find(
  (n) => n.id === U_
) ?? bu[0], F_ = N_?.trigger ?? Vs.trigger, is = Zi.modes, xs = Zi.room_presets, Ms = Zi.reverb_levels, O_ = Zi.outdoor_times, B_ = Zi.outdoor_level, z_ = Zi.sfx_presets, Ss = Zi.sfx_levels, pi = {
  space_mode: "room",
  room_preset: "medium_room",
  outdoor_time: "day",
  sfx_preset: "dual_delay",
  effect_level: "mid",
  sfx_level: "low",
  source_type: "male spoken voice"
};
function H_(n) {
  return n.space_mode === "dry" ? "dry" : n.space_mode === "outside" ? n.outdoor_time === "night" ? "outdoor_night" : "outdoor_day" : n.space_mode === "sfx" ? z_.includes(n.sfx_preset) ? n.sfx_preset : "dual_delay" : xs.includes(n.room_preset) ? n.room_preset : "medium_room";
}
function vc(n) {
  return Jf[H_(n)];
}
function Qf(n) {
  return n.space_mode === "outside" ? B_ : n.space_mode === "sfx" ? Ss.includes(n.sfx_level) ? n.sfx_level : Ss[0] : Ms.includes(n.effect_level) ? n.effect_level : "mid";
}
function V_(n) {
  const e = vc(n), t = dr[Qf(n)], [i, s, r] = e.dimensions_m;
  return {
    rt60: e.estimated_rt60,
    predelay_ms: e.estimated_predelay_ms,
    volume_m3: i * s * r,
    visual_amount: n.space_mode === "dry" ? 0 : t.visual_amount
  };
}
function k_(n) {
  const e = vc(n);
  if (e.mode === "dry")
    return `${n.source_type}, close-miked dry reference, no reverb, no background ambience`;
  const t = dr[Qf(n)], i = e.caption_tail ? `, ${e.caption_tail}` : "", s = n.source_type?.trim(), r = s ? `${s} ` : "";
  return `${F_} ${r}${e.caption_where}, ${t.caption_word} ${e.caption_character}${i}`;
}
function G_(n) {
  return k_({ ...n, source_type: "" });
}
function W_(n, e, t = !0) {
  const i = String(n ?? "").trim(), s = String(e ?? "").trim();
  if (!t) return i;
  if (!i) return s;
  if (!s) return i;
  const r = /[,.;:]$/.test(i) ? " " : ", ";
  return `${i}${r}${s}`;
}
const xc = "180", Ds = { ROTATE: 0, DOLLY: 1, PAN: 2 }, ys = { ROTATE: 0, PAN: 1, DOLLY_PAN: 2, DOLLY_ROTATE: 3 }, X_ = 0, Au = 1, Y_ = 2, ed = 1, q_ = 2, jn = 3, yi = 0, Wt = 1, Qn = 2, xi = 0, Ls = 1, wu = 2, Ru = 3, Cu = 4, j_ = 5, zi = 100, K_ = 101, $_ = 102, Z_ = 103, J_ = 104, Q_ = 200, eg = 201, tg = 202, ng = 203, fl = 204, dl = 205, ig = 206, sg = 207, rg = 208, og = 209, ag = 210, lg = 211, cg = 212, ug = 213, hg = 214, pl = 0, ml = 1, _l = 2, Ns = 3, gl = 4, vl = 5, xl = 6, Ml = 7, td = 0, fg = 1, dg = 2, Mi = 0, pg = 1, mg = 2, _g = 3, nd = 4, gg = 5, vg = 6, xg = 7, id = 300, Fs = 301, Os = 302, Sl = 303, yl = 304, ta = 306, El = 1e3, Vi = 1001, Tl = 1002, yn = 1003, Mg = 1004, kr = 1005, Un = 1006, xa = 1007, ki = 1008, Bn = 1009, sd = 1010, rd = 1011, br = 1012, Mc = 1013, Yi = 1014, ei = 1015, Ir = 1016, Sc = 1017, yc = 1018, Ar = 1020, od = 35902, ad = 35899, ld = 1021, cd = 1022, xn = 1023, wr = 1026, Rr = 1027, ud = 1028, Ec = 1029, hd = 1030, Tc = 1031, bc = 1033, So = 33776, yo = 33777, Eo = 33778, To = 33779, bl = 35840, Al = 35841, wl = 35842, Rl = 35843, Cl = 36196, Pl = 37492, Dl = 37496, Ll = 37808, Il = 37809, Ul = 37810, Nl = 37811, Fl = 37812, Ol = 37813, Bl = 37814, zl = 37815, Hl = 37816, Vl = 37817, kl = 37818, Gl = 37819, Wl = 37820, Xl = 37821, Yl = 36492, ql = 36494, jl = 36495, Kl = 36283, $l = 36284, Zl = 36285, Jl = 36286, Sg = 3200, yg = 3201, fd = 0, Eg = 1, vi = "", sn = "srgb", Bs = "srgb-linear", Bo = "linear", ot = "srgb", ss = 7680, Pu = 519, Tg = 512, bg = 513, Ag = 514, dd = 515, wg = 516, Rg = 517, Cg = 518, Pg = 519, Du = 35044, Lu = "300 es", Nn = 2e3, zo = 2001;
class Ji {
  /**
   * Adds the given event listener to the given event type.
   *
   * @param {string} type - The type of event to listen to.
   * @param {Function} listener - The function that gets called when the event is fired.
   */
  addEventListener(e, t) {
    this._listeners === void 0 && (this._listeners = {});
    const i = this._listeners;
    i[e] === void 0 && (i[e] = []), i[e].indexOf(t) === -1 && i[e].push(t);
  }
  /**
   * Returns `true` if the given event listener has been added to the given event type.
   *
   * @param {string} type - The type of event.
   * @param {Function} listener - The listener to check.
   * @return {boolean} Whether the given event listener has been added to the given event type.
   */
  hasEventListener(e, t) {
    const i = this._listeners;
    return i === void 0 ? !1 : i[e] !== void 0 && i[e].indexOf(t) !== -1;
  }
  /**
   * Removes the given event listener from the given event type.
   *
   * @param {string} type - The type of event.
   * @param {Function} listener - The listener to remove.
   */
  removeEventListener(e, t) {
    const i = this._listeners;
    if (i === void 0) return;
    const s = i[e];
    if (s !== void 0) {
      const r = s.indexOf(t);
      r !== -1 && s.splice(r, 1);
    }
  }
  /**
   * Dispatches an event object.
   *
   * @param {Object} event - The event that gets fired.
   */
  dispatchEvent(e) {
    const t = this._listeners;
    if (t === void 0) return;
    const i = t[e.type];
    if (i !== void 0) {
      e.target = this;
      const s = i.slice(0);
      for (let r = 0, o = s.length; r < o; r++)
        s[r].call(this, e);
      e.target = null;
    }
  }
}
const Pt = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "0a", "0b", "0c", "0d", "0e", "0f", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "1a", "1b", "1c", "1d", "1e", "1f", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "2a", "2b", "2c", "2d", "2e", "2f", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "3a", "3b", "3c", "3d", "3e", "3f", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "4a", "4b", "4c", "4d", "4e", "4f", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "5a", "5b", "5c", "5d", "5e", "5f", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "6a", "6b", "6c", "6d", "6e", "6f", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "7a", "7b", "7c", "7d", "7e", "7f", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "8a", "8b", "8c", "8d", "8e", "8f", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "9a", "9b", "9c", "9d", "9e", "9f", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "aa", "ab", "ac", "ad", "ae", "af", "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "ba", "bb", "bc", "bd", "be", "bf", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "ca", "cb", "cc", "cd", "ce", "cf", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "da", "db", "dc", "dd", "de", "df", "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "ea", "eb", "ec", "ed", "ee", "ef", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "fa", "fb", "fc", "fd", "fe", "ff"], pr = Math.PI / 180, Ql = 180 / Math.PI;
function Ur() {
  const n = Math.random() * 4294967295 | 0, e = Math.random() * 4294967295 | 0, t = Math.random() * 4294967295 | 0, i = Math.random() * 4294967295 | 0;
  return (Pt[n & 255] + Pt[n >> 8 & 255] + Pt[n >> 16 & 255] + Pt[n >> 24 & 255] + "-" + Pt[e & 255] + Pt[e >> 8 & 255] + "-" + Pt[e >> 16 & 15 | 64] + Pt[e >> 24 & 255] + "-" + Pt[t & 63 | 128] + Pt[t >> 8 & 255] + "-" + Pt[t >> 16 & 255] + Pt[t >> 24 & 255] + Pt[i & 255] + Pt[i >> 8 & 255] + Pt[i >> 16 & 255] + Pt[i >> 24 & 255]).toLowerCase();
}
function Ke(n, e, t) {
  return Math.max(e, Math.min(t, n));
}
function Dg(n, e) {
  return (n % e + e) % e;
}
function Ma(n, e, t) {
  return (1 - t) * n + t * e;
}
function Ks(n, e) {
  switch (e.constructor) {
    case Float32Array:
      return n;
    case Uint32Array:
      return n / 4294967295;
    case Uint16Array:
      return n / 65535;
    case Uint8Array:
      return n / 255;
    case Int32Array:
      return Math.max(n / 2147483647, -1);
    case Int16Array:
      return Math.max(n / 32767, -1);
    case Int8Array:
      return Math.max(n / 127, -1);
    default:
      throw new Error("Invalid component type.");
  }
}
function Yt(n, e) {
  switch (e.constructor) {
    case Float32Array:
      return n;
    case Uint32Array:
      return Math.round(n * 4294967295);
    case Uint16Array:
      return Math.round(n * 65535);
    case Uint8Array:
      return Math.round(n * 255);
    case Int32Array:
      return Math.round(n * 2147483647);
    case Int16Array:
      return Math.round(n * 32767);
    case Int8Array:
      return Math.round(n * 127);
    default:
      throw new Error("Invalid component type.");
  }
}
const Lg = {
  DEG2RAD: pr
};
class Ve {
  /**
   * Constructs a new 2D vector.
   *
   * @param {number} [x=0] - The x value of this vector.
   * @param {number} [y=0] - The y value of this vector.
   */
  constructor(e = 0, t = 0) {
    Ve.prototype.isVector2 = !0, this.x = e, this.y = t;
  }
  /**
   * Alias for {@link Vector2#x}.
   *
   * @type {number}
   */
  get width() {
    return this.x;
  }
  set width(e) {
    this.x = e;
  }
  /**
   * Alias for {@link Vector2#y}.
   *
   * @type {number}
   */
  get height() {
    return this.y;
  }
  set height(e) {
    this.y = e;
  }
  /**
   * Sets the vector components.
   *
   * @param {number} x - The value of the x component.
   * @param {number} y - The value of the y component.
   * @return {Vector2} A reference to this vector.
   */
  set(e, t) {
    return this.x = e, this.y = t, this;
  }
  /**
   * Sets the vector components to the same value.
   *
   * @param {number} scalar - The value to set for all vector components.
   * @return {Vector2} A reference to this vector.
   */
  setScalar(e) {
    return this.x = e, this.y = e, this;
  }
  /**
   * Sets the vector's x component to the given value
   *
   * @param {number} x - The value to set.
   * @return {Vector2} A reference to this vector.
   */
  setX(e) {
    return this.x = e, this;
  }
  /**
   * Sets the vector's y component to the given value
   *
   * @param {number} y - The value to set.
   * @return {Vector2} A reference to this vector.
   */
  setY(e) {
    return this.y = e, this;
  }
  /**
   * Allows to set a vector component with an index.
   *
   * @param {number} index - The component index. `0` equals to x, `1` equals to y.
   * @param {number} value - The value to set.
   * @return {Vector2} A reference to this vector.
   */
  setComponent(e, t) {
    switch (e) {
      case 0:
        this.x = t;
        break;
      case 1:
        this.y = t;
        break;
      default:
        throw new Error("index is out of range: " + e);
    }
    return this;
  }
  /**
   * Returns the value of the vector component which matches the given index.
   *
   * @param {number} index - The component index. `0` equals to x, `1` equals to y.
   * @return {number} A vector component value.
   */
  getComponent(e) {
    switch (e) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      default:
        throw new Error("index is out of range: " + e);
    }
  }
  /**
   * Returns a new vector with copied values from this instance.
   *
   * @return {Vector2} A clone of this instance.
   */
  clone() {
    return new this.constructor(this.x, this.y);
  }
  /**
   * Copies the values of the given vector to this instance.
   *
   * @param {Vector2} v - The vector to copy.
   * @return {Vector2} A reference to this vector.
   */
  copy(e) {
    return this.x = e.x, this.y = e.y, this;
  }
  /**
   * Adds the given vector to this instance.
   *
   * @param {Vector2} v - The vector to add.
   * @return {Vector2} A reference to this vector.
   */
  add(e) {
    return this.x += e.x, this.y += e.y, this;
  }
  /**
   * Adds the given scalar value to all components of this instance.
   *
   * @param {number} s - The scalar to add.
   * @return {Vector2} A reference to this vector.
   */
  addScalar(e) {
    return this.x += e, this.y += e, this;
  }
  /**
   * Adds the given vectors and stores the result in this instance.
   *
   * @param {Vector2} a - The first vector.
   * @param {Vector2} b - The second vector.
   * @return {Vector2} A reference to this vector.
   */
  addVectors(e, t) {
    return this.x = e.x + t.x, this.y = e.y + t.y, this;
  }
  /**
   * Adds the given vector scaled by the given factor to this instance.
   *
   * @param {Vector2} v - The vector.
   * @param {number} s - The factor that scales `v`.
   * @return {Vector2} A reference to this vector.
   */
  addScaledVector(e, t) {
    return this.x += e.x * t, this.y += e.y * t, this;
  }
  /**
   * Subtracts the given vector from this instance.
   *
   * @param {Vector2} v - The vector to subtract.
   * @return {Vector2} A reference to this vector.
   */
  sub(e) {
    return this.x -= e.x, this.y -= e.y, this;
  }
  /**
   * Subtracts the given scalar value from all components of this instance.
   *
   * @param {number} s - The scalar to subtract.
   * @return {Vector2} A reference to this vector.
   */
  subScalar(e) {
    return this.x -= e, this.y -= e, this;
  }
  /**
   * Subtracts the given vectors and stores the result in this instance.
   *
   * @param {Vector2} a - The first vector.
   * @param {Vector2} b - The second vector.
   * @return {Vector2} A reference to this vector.
   */
  subVectors(e, t) {
    return this.x = e.x - t.x, this.y = e.y - t.y, this;
  }
  /**
   * Multiplies the given vector with this instance.
   *
   * @param {Vector2} v - The vector to multiply.
   * @return {Vector2} A reference to this vector.
   */
  multiply(e) {
    return this.x *= e.x, this.y *= e.y, this;
  }
  /**
   * Multiplies the given scalar value with all components of this instance.
   *
   * @param {number} scalar - The scalar to multiply.
   * @return {Vector2} A reference to this vector.
   */
  multiplyScalar(e) {
    return this.x *= e, this.y *= e, this;
  }
  /**
   * Divides this instance by the given vector.
   *
   * @param {Vector2} v - The vector to divide.
   * @return {Vector2} A reference to this vector.
   */
  divide(e) {
    return this.x /= e.x, this.y /= e.y, this;
  }
  /**
   * Divides this vector by the given scalar.
   *
   * @param {number} scalar - The scalar to divide.
   * @return {Vector2} A reference to this vector.
   */
  divideScalar(e) {
    return this.multiplyScalar(1 / e);
  }
  /**
   * Multiplies this vector (with an implicit 1 as the 3rd component) by
   * the given 3x3 matrix.
   *
   * @param {Matrix3} m - The matrix to apply.
   * @return {Vector2} A reference to this vector.
   */
  applyMatrix3(e) {
    const t = this.x, i = this.y, s = e.elements;
    return this.x = s[0] * t + s[3] * i + s[6], this.y = s[1] * t + s[4] * i + s[7], this;
  }
  /**
   * If this vector's x or y value is greater than the given vector's x or y
   * value, replace that value with the corresponding min value.
   *
   * @param {Vector2} v - The vector.
   * @return {Vector2} A reference to this vector.
   */
  min(e) {
    return this.x = Math.min(this.x, e.x), this.y = Math.min(this.y, e.y), this;
  }
  /**
   * If this vector's x or y value is less than the given vector's x or y
   * value, replace that value with the corresponding max value.
   *
   * @param {Vector2} v - The vector.
   * @return {Vector2} A reference to this vector.
   */
  max(e) {
    return this.x = Math.max(this.x, e.x), this.y = Math.max(this.y, e.y), this;
  }
  /**
   * If this vector's x or y value is greater than the max vector's x or y
   * value, it is replaced by the corresponding value.
   * If this vector's x or y value is less than the min vector's x or y value,
   * it is replaced by the corresponding value.
   *
   * @param {Vector2} min - The minimum x and y values.
   * @param {Vector2} max - The maximum x and y values in the desired range.
   * @return {Vector2} A reference to this vector.
   */
  clamp(e, t) {
    return this.x = Ke(this.x, e.x, t.x), this.y = Ke(this.y, e.y, t.y), this;
  }
  /**
   * If this vector's x or y values are greater than the max value, they are
   * replaced by the max value.
   * If this vector's x or y values are less than the min value, they are
   * replaced by the min value.
   *
   * @param {number} minVal - The minimum value the components will be clamped to.
   * @param {number} maxVal - The maximum value the components will be clamped to.
   * @return {Vector2} A reference to this vector.
   */
  clampScalar(e, t) {
    return this.x = Ke(this.x, e, t), this.y = Ke(this.y, e, t), this;
  }
  /**
   * If this vector's length is greater than the max value, it is replaced by
   * the max value.
   * If this vector's length is less than the min value, it is replaced by the
   * min value.
   *
   * @param {number} min - The minimum value the vector length will be clamped to.
   * @param {number} max - The maximum value the vector length will be clamped to.
   * @return {Vector2} A reference to this vector.
   */
  clampLength(e, t) {
    const i = this.length();
    return this.divideScalar(i || 1).multiplyScalar(Ke(i, e, t));
  }
  /**
   * The components of this vector are rounded down to the nearest integer value.
   *
   * @return {Vector2} A reference to this vector.
   */
  floor() {
    return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this;
  }
  /**
   * The components of this vector are rounded up to the nearest integer value.
   *
   * @return {Vector2} A reference to this vector.
   */
  ceil() {
    return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this;
  }
  /**
   * The components of this vector are rounded to the nearest integer value
   *
   * @return {Vector2} A reference to this vector.
   */
  round() {
    return this.x = Math.round(this.x), this.y = Math.round(this.y), this;
  }
  /**
   * The components of this vector are rounded towards zero (up if negative,
   * down if positive) to an integer value.
   *
   * @return {Vector2} A reference to this vector.
   */
  roundToZero() {
    return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this;
  }
  /**
   * Inverts this vector - i.e. sets x = -x and y = -y.
   *
   * @return {Vector2} A reference to this vector.
   */
  negate() {
    return this.x = -this.x, this.y = -this.y, this;
  }
  /**
   * Calculates the dot product of the given vector with this instance.
   *
   * @param {Vector2} v - The vector to compute the dot product with.
   * @return {number} The result of the dot product.
   */
  dot(e) {
    return this.x * e.x + this.y * e.y;
  }
  /**
   * Calculates the cross product of the given vector with this instance.
   *
   * @param {Vector2} v - The vector to compute the cross product with.
   * @return {number} The result of the cross product.
   */
  cross(e) {
    return this.x * e.y - this.y * e.x;
  }
  /**
   * Computes the square of the Euclidean length (straight-line length) from
   * (0, 0) to (x, y). If you are comparing the lengths of vectors, you should
   * compare the length squared instead as it is slightly more efficient to calculate.
   *
   * @return {number} The square length of this vector.
   */
  lengthSq() {
    return this.x * this.x + this.y * this.y;
  }
  /**
   * Computes the  Euclidean length (straight-line length) from (0, 0) to (x, y).
   *
   * @return {number} The length of this vector.
   */
  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y);
  }
  /**
   * Computes the Manhattan length of this vector.
   *
   * @return {number} The length of this vector.
   */
  manhattanLength() {
    return Math.abs(this.x) + Math.abs(this.y);
  }
  /**
   * Converts this vector to a unit vector - that is, sets it equal to a vector
   * with the same direction as this one, but with a vector length of `1`.
   *
   * @return {Vector2} A reference to this vector.
   */
  normalize() {
    return this.divideScalar(this.length() || 1);
  }
  /**
   * Computes the angle in radians of this vector with respect to the positive x-axis.
   *
   * @return {number} The angle in radians.
   */
  angle() {
    return Math.atan2(-this.y, -this.x) + Math.PI;
  }
  /**
   * Returns the angle between the given vector and this instance in radians.
   *
   * @param {Vector2} v - The vector to compute the angle with.
   * @return {number} The angle in radians.
   */
  angleTo(e) {
    const t = Math.sqrt(this.lengthSq() * e.lengthSq());
    if (t === 0) return Math.PI / 2;
    const i = this.dot(e) / t;
    return Math.acos(Ke(i, -1, 1));
  }
  /**
   * Computes the distance from the given vector to this instance.
   *
   * @param {Vector2} v - The vector to compute the distance to.
   * @return {number} The distance.
   */
  distanceTo(e) {
    return Math.sqrt(this.distanceToSquared(e));
  }
  /**
   * Computes the squared distance from the given vector to this instance.
   * If you are just comparing the distance with another distance, you should compare
   * the distance squared instead as it is slightly more efficient to calculate.
   *
   * @param {Vector2} v - The vector to compute the squared distance to.
   * @return {number} The squared distance.
   */
  distanceToSquared(e) {
    const t = this.x - e.x, i = this.y - e.y;
    return t * t + i * i;
  }
  /**
   * Computes the Manhattan distance from the given vector to this instance.
   *
   * @param {Vector2} v - The vector to compute the Manhattan distance to.
   * @return {number} The Manhattan distance.
   */
  manhattanDistanceTo(e) {
    return Math.abs(this.x - e.x) + Math.abs(this.y - e.y);
  }
  /**
   * Sets this vector to a vector with the same direction as this one, but
   * with the specified length.
   *
   * @param {number} length - The new length of this vector.
   * @return {Vector2} A reference to this vector.
   */
  setLength(e) {
    return this.normalize().multiplyScalar(e);
  }
  /**
   * Linearly interpolates between the given vector and this instance, where
   * alpha is the percent distance along the line - alpha = 0 will be this
   * vector, and alpha = 1 will be the given one.
   *
   * @param {Vector2} v - The vector to interpolate towards.
   * @param {number} alpha - The interpolation factor, typically in the closed interval `[0, 1]`.
   * @return {Vector2} A reference to this vector.
   */
  lerp(e, t) {
    return this.x += (e.x - this.x) * t, this.y += (e.y - this.y) * t, this;
  }
  /**
   * Linearly interpolates between the given vectors, where alpha is the percent
   * distance along the line - alpha = 0 will be first vector, and alpha = 1 will
   * be the second one. The result is stored in this instance.
   *
   * @param {Vector2} v1 - The first vector.
   * @param {Vector2} v2 - The second vector.
   * @param {number} alpha - The interpolation factor, typically in the closed interval `[0, 1]`.
   * @return {Vector2} A reference to this vector.
   */
  lerpVectors(e, t, i) {
    return this.x = e.x + (t.x - e.x) * i, this.y = e.y + (t.y - e.y) * i, this;
  }
  /**
   * Returns `true` if this vector is equal with the given one.
   *
   * @param {Vector2} v - The vector to test for equality.
   * @return {boolean} Whether this vector is equal with the given one.
   */
  equals(e) {
    return e.x === this.x && e.y === this.y;
  }
  /**
   * Sets this vector's x value to be `array[ offset ]` and y
   * value to be `array[ offset + 1 ]`.
   *
   * @param {Array<number>} array - An array holding the vector component values.
   * @param {number} [offset=0] - The offset into the array.
   * @return {Vector2} A reference to this vector.
   */
  fromArray(e, t = 0) {
    return this.x = e[t], this.y = e[t + 1], this;
  }
  /**
   * Writes the components of this vector to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the vector components.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The vector components.
   */
  toArray(e = [], t = 0) {
    return e[t] = this.x, e[t + 1] = this.y, e;
  }
  /**
   * Sets the components of this vector from the given buffer attribute.
   *
   * @param {BufferAttribute} attribute - The buffer attribute holding vector data.
   * @param {number} index - The index into the attribute.
   * @return {Vector2} A reference to this vector.
   */
  fromBufferAttribute(e, t) {
    return this.x = e.getX(t), this.y = e.getY(t), this;
  }
  /**
   * Rotates this vector around the given center by the given angle.
   *
   * @param {Vector2} center - The point around which to rotate.
   * @param {number} angle - The angle to rotate, in radians.
   * @return {Vector2} A reference to this vector.
   */
  rotateAround(e, t) {
    const i = Math.cos(t), s = Math.sin(t), r = this.x - e.x, o = this.y - e.y;
    return this.x = r * i - o * s + e.x, this.y = r * s + o * i + e.y, this;
  }
  /**
   * Sets each component of this vector to a pseudo-random value between `0` and
   * `1`, excluding `1`.
   *
   * @return {Vector2} A reference to this vector.
   */
  random() {
    return this.x = Math.random(), this.y = Math.random(), this;
  }
  *[Symbol.iterator]() {
    yield this.x, yield this.y;
  }
}
class qi {
  /**
   * Constructs a new quaternion.
   *
   * @param {number} [x=0] - The x value of this quaternion.
   * @param {number} [y=0] - The y value of this quaternion.
   * @param {number} [z=0] - The z value of this quaternion.
   * @param {number} [w=1] - The w value of this quaternion.
   */
  constructor(e = 0, t = 0, i = 0, s = 1) {
    this.isQuaternion = !0, this._x = e, this._y = t, this._z = i, this._w = s;
  }
  /**
   * Interpolates between two quaternions via SLERP. This implementation assumes the
   * quaternion data are managed  in flat arrays.
   *
   * @param {Array<number>} dst - The destination array.
   * @param {number} dstOffset - An offset into the destination array.
   * @param {Array<number>} src0 - The source array of the first quaternion.
   * @param {number} srcOffset0 - An offset into the first source array.
   * @param {Array<number>} src1 -  The source array of the second quaternion.
   * @param {number} srcOffset1 - An offset into the second source array.
   * @param {number} t - The interpolation factor in the range `[0,1]`.
   * @see {@link Quaternion#slerp}
   */
  static slerpFlat(e, t, i, s, r, o, a) {
    let l = i[s + 0], c = i[s + 1], u = i[s + 2], h = i[s + 3];
    const f = r[o + 0], p = r[o + 1], v = r[o + 2], x = r[o + 3];
    if (a === 0) {
      e[t + 0] = l, e[t + 1] = c, e[t + 2] = u, e[t + 3] = h;
      return;
    }
    if (a === 1) {
      e[t + 0] = f, e[t + 1] = p, e[t + 2] = v, e[t + 3] = x;
      return;
    }
    if (h !== x || l !== f || c !== p || u !== v) {
      let m = 1 - a;
      const d = l * f + c * p + u * v + h * x, b = d >= 0 ? 1 : -1, A = 1 - d * d;
      if (A > Number.EPSILON) {
        const C = Math.sqrt(A), w = Math.atan2(C, d * b);
        m = Math.sin(m * w) / C, a = Math.sin(a * w) / C;
      }
      const M = a * b;
      if (l = l * m + f * M, c = c * m + p * M, u = u * m + v * M, h = h * m + x * M, m === 1 - a) {
        const C = 1 / Math.sqrt(l * l + c * c + u * u + h * h);
        l *= C, c *= C, u *= C, h *= C;
      }
    }
    e[t] = l, e[t + 1] = c, e[t + 2] = u, e[t + 3] = h;
  }
  /**
   * Multiplies two quaternions. This implementation assumes the quaternion data are managed
   * in flat arrays.
   *
   * @param {Array<number>} dst - The destination array.
   * @param {number} dstOffset - An offset into the destination array.
   * @param {Array<number>} src0 - The source array of the first quaternion.
   * @param {number} srcOffset0 - An offset into the first source array.
   * @param {Array<number>} src1 -  The source array of the second quaternion.
   * @param {number} srcOffset1 - An offset into the second source array.
   * @return {Array<number>} The destination array.
   * @see {@link Quaternion#multiplyQuaternions}.
   */
  static multiplyQuaternionsFlat(e, t, i, s, r, o) {
    const a = i[s], l = i[s + 1], c = i[s + 2], u = i[s + 3], h = r[o], f = r[o + 1], p = r[o + 2], v = r[o + 3];
    return e[t] = a * v + u * h + l * p - c * f, e[t + 1] = l * v + u * f + c * h - a * p, e[t + 2] = c * v + u * p + a * f - l * h, e[t + 3] = u * v - a * h - l * f - c * p, e;
  }
  /**
   * The x value of this quaternion.
   *
   * @type {number}
   * @default 0
   */
  get x() {
    return this._x;
  }
  set x(e) {
    this._x = e, this._onChangeCallback();
  }
  /**
   * The y value of this quaternion.
   *
   * @type {number}
   * @default 0
   */
  get y() {
    return this._y;
  }
  set y(e) {
    this._y = e, this._onChangeCallback();
  }
  /**
   * The z value of this quaternion.
   *
   * @type {number}
   * @default 0
   */
  get z() {
    return this._z;
  }
  set z(e) {
    this._z = e, this._onChangeCallback();
  }
  /**
   * The w value of this quaternion.
   *
   * @type {number}
   * @default 1
   */
  get w() {
    return this._w;
  }
  set w(e) {
    this._w = e, this._onChangeCallback();
  }
  /**
   * Sets the quaternion components.
   *
   * @param {number} x - The x value of this quaternion.
   * @param {number} y - The y value of this quaternion.
   * @param {number} z - The z value of this quaternion.
   * @param {number} w - The w value of this quaternion.
   * @return {Quaternion} A reference to this quaternion.
   */
  set(e, t, i, s) {
    return this._x = e, this._y = t, this._z = i, this._w = s, this._onChangeCallback(), this;
  }
  /**
   * Returns a new quaternion with copied values from this instance.
   *
   * @return {Quaternion} A clone of this instance.
   */
  clone() {
    return new this.constructor(this._x, this._y, this._z, this._w);
  }
  /**
   * Copies the values of the given quaternion to this instance.
   *
   * @param {Quaternion} quaternion - The quaternion to copy.
   * @return {Quaternion} A reference to this quaternion.
   */
  copy(e) {
    return this._x = e.x, this._y = e.y, this._z = e.z, this._w = e.w, this._onChangeCallback(), this;
  }
  /**
   * Sets this quaternion from the rotation specified by the given
   * Euler angles.
   *
   * @param {Euler} euler - The Euler angles.
   * @param {boolean} [update=true] - Whether the internal `onChange` callback should be executed or not.
   * @return {Quaternion} A reference to this quaternion.
   */
  setFromEuler(e, t = !0) {
    const i = e._x, s = e._y, r = e._z, o = e._order, a = Math.cos, l = Math.sin, c = a(i / 2), u = a(s / 2), h = a(r / 2), f = l(i / 2), p = l(s / 2), v = l(r / 2);
    switch (o) {
      case "XYZ":
        this._x = f * u * h + c * p * v, this._y = c * p * h - f * u * v, this._z = c * u * v + f * p * h, this._w = c * u * h - f * p * v;
        break;
      case "YXZ":
        this._x = f * u * h + c * p * v, this._y = c * p * h - f * u * v, this._z = c * u * v - f * p * h, this._w = c * u * h + f * p * v;
        break;
      case "ZXY":
        this._x = f * u * h - c * p * v, this._y = c * p * h + f * u * v, this._z = c * u * v + f * p * h, this._w = c * u * h - f * p * v;
        break;
      case "ZYX":
        this._x = f * u * h - c * p * v, this._y = c * p * h + f * u * v, this._z = c * u * v - f * p * h, this._w = c * u * h + f * p * v;
        break;
      case "YZX":
        this._x = f * u * h + c * p * v, this._y = c * p * h + f * u * v, this._z = c * u * v - f * p * h, this._w = c * u * h - f * p * v;
        break;
      case "XZY":
        this._x = f * u * h - c * p * v, this._y = c * p * h - f * u * v, this._z = c * u * v + f * p * h, this._w = c * u * h + f * p * v;
        break;
      default:
        console.warn("THREE.Quaternion: .setFromEuler() encountered an unknown order: " + o);
    }
    return t === !0 && this._onChangeCallback(), this;
  }
  /**
   * Sets this quaternion from the given axis and angle.
   *
   * @param {Vector3} axis - The normalized axis.
   * @param {number} angle - The angle in radians.
   * @return {Quaternion} A reference to this quaternion.
   */
  setFromAxisAngle(e, t) {
    const i = t / 2, s = Math.sin(i);
    return this._x = e.x * s, this._y = e.y * s, this._z = e.z * s, this._w = Math.cos(i), this._onChangeCallback(), this;
  }
  /**
   * Sets this quaternion from the given rotation matrix.
   *
   * @param {Matrix4} m - A 4x4 matrix of which the upper 3x3 of matrix is a pure rotation matrix (i.e. unscaled).
   * @return {Quaternion} A reference to this quaternion.
   */
  setFromRotationMatrix(e) {
    const t = e.elements, i = t[0], s = t[4], r = t[8], o = t[1], a = t[5], l = t[9], c = t[2], u = t[6], h = t[10], f = i + a + h;
    if (f > 0) {
      const p = 0.5 / Math.sqrt(f + 1);
      this._w = 0.25 / p, this._x = (u - l) * p, this._y = (r - c) * p, this._z = (o - s) * p;
    } else if (i > a && i > h) {
      const p = 2 * Math.sqrt(1 + i - a - h);
      this._w = (u - l) / p, this._x = 0.25 * p, this._y = (s + o) / p, this._z = (r + c) / p;
    } else if (a > h) {
      const p = 2 * Math.sqrt(1 + a - i - h);
      this._w = (r - c) / p, this._x = (s + o) / p, this._y = 0.25 * p, this._z = (l + u) / p;
    } else {
      const p = 2 * Math.sqrt(1 + h - i - a);
      this._w = (o - s) / p, this._x = (r + c) / p, this._y = (l + u) / p, this._z = 0.25 * p;
    }
    return this._onChangeCallback(), this;
  }
  /**
   * Sets this quaternion to the rotation required to rotate the direction vector
   * `vFrom` to the direction vector `vTo`.
   *
   * @param {Vector3} vFrom - The first (normalized) direction vector.
   * @param {Vector3} vTo - The second (normalized) direction vector.
   * @return {Quaternion} A reference to this quaternion.
   */
  setFromUnitVectors(e, t) {
    let i = e.dot(t) + 1;
    return i < 1e-8 ? (i = 0, Math.abs(e.x) > Math.abs(e.z) ? (this._x = -e.y, this._y = e.x, this._z = 0, this._w = i) : (this._x = 0, this._y = -e.z, this._z = e.y, this._w = i)) : (this._x = e.y * t.z - e.z * t.y, this._y = e.z * t.x - e.x * t.z, this._z = e.x * t.y - e.y * t.x, this._w = i), this.normalize();
  }
  /**
   * Returns the angle between this quaternion and the given one in radians.
   *
   * @param {Quaternion} q - The quaternion to compute the angle with.
   * @return {number} The angle in radians.
   */
  angleTo(e) {
    return 2 * Math.acos(Math.abs(Ke(this.dot(e), -1, 1)));
  }
  /**
   * Rotates this quaternion by a given angular step to the given quaternion.
   * The method ensures that the final quaternion will not overshoot `q`.
   *
   * @param {Quaternion} q - The target quaternion.
   * @param {number} step - The angular step in radians.
   * @return {Quaternion} A reference to this quaternion.
   */
  rotateTowards(e, t) {
    const i = this.angleTo(e);
    if (i === 0) return this;
    const s = Math.min(1, t / i);
    return this.slerp(e, s), this;
  }
  /**
   * Sets this quaternion to the identity quaternion; that is, to the
   * quaternion that represents "no rotation".
   *
   * @return {Quaternion} A reference to this quaternion.
   */
  identity() {
    return this.set(0, 0, 0, 1);
  }
  /**
   * Inverts this quaternion via {@link Quaternion#conjugate}. The
   * quaternion is assumed to have unit length.
   *
   * @return {Quaternion} A reference to this quaternion.
   */
  invert() {
    return this.conjugate();
  }
  /**
   * Returns the rotational conjugate of this quaternion. The conjugate of a
   * quaternion represents the same rotation in the opposite direction about
   * the rotational axis.
   *
   * @return {Quaternion} A reference to this quaternion.
   */
  conjugate() {
    return this._x *= -1, this._y *= -1, this._z *= -1, this._onChangeCallback(), this;
  }
  /**
   * Calculates the dot product of this quaternion and the given one.
   *
   * @param {Quaternion} v - The quaternion to compute the dot product with.
   * @return {number} The result of the dot product.
   */
  dot(e) {
    return this._x * e._x + this._y * e._y + this._z * e._z + this._w * e._w;
  }
  /**
   * Computes the squared Euclidean length (straight-line length) of this quaternion,
   * considered as a 4 dimensional vector. This can be useful if you are comparing the
   * lengths of two quaternions, as this is a slightly more efficient calculation than
   * {@link Quaternion#length}.
   *
   * @return {number} The squared Euclidean length.
   */
  lengthSq() {
    return this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w;
  }
  /**
   * Computes the Euclidean length (straight-line length) of this quaternion,
   * considered as a 4 dimensional vector.
   *
   * @return {number} The Euclidean length.
   */
  length() {
    return Math.sqrt(this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w);
  }
  /**
   * Normalizes this quaternion - that is, calculated the quaternion that performs
   * the same rotation as this one, but has a length equal to `1`.
   *
   * @return {Quaternion} A reference to this quaternion.
   */
  normalize() {
    let e = this.length();
    return e === 0 ? (this._x = 0, this._y = 0, this._z = 0, this._w = 1) : (e = 1 / e, this._x = this._x * e, this._y = this._y * e, this._z = this._z * e, this._w = this._w * e), this._onChangeCallback(), this;
  }
  /**
   * Multiplies this quaternion by the given one.
   *
   * @param {Quaternion} q - The quaternion.
   * @return {Quaternion} A reference to this quaternion.
   */
  multiply(e) {
    return this.multiplyQuaternions(this, e);
  }
  /**
   * Pre-multiplies this quaternion by the given one.
   *
   * @param {Quaternion} q - The quaternion.
   * @return {Quaternion} A reference to this quaternion.
   */
  premultiply(e) {
    return this.multiplyQuaternions(e, this);
  }
  /**
   * Multiplies the given quaternions and stores the result in this instance.
   *
   * @param {Quaternion} a - The first quaternion.
   * @param {Quaternion} b - The second quaternion.
   * @return {Quaternion} A reference to this quaternion.
   */
  multiplyQuaternions(e, t) {
    const i = e._x, s = e._y, r = e._z, o = e._w, a = t._x, l = t._y, c = t._z, u = t._w;
    return this._x = i * u + o * a + s * c - r * l, this._y = s * u + o * l + r * a - i * c, this._z = r * u + o * c + i * l - s * a, this._w = o * u - i * a - s * l - r * c, this._onChangeCallback(), this;
  }
  /**
   * Performs a spherical linear interpolation between quaternions.
   *
   * @param {Quaternion} qb - The target quaternion.
   * @param {number} t - The interpolation factor in the closed interval `[0, 1]`.
   * @return {Quaternion} A reference to this quaternion.
   */
  slerp(e, t) {
    if (t === 0) return this;
    if (t === 1) return this.copy(e);
    const i = this._x, s = this._y, r = this._z, o = this._w;
    let a = o * e._w + i * e._x + s * e._y + r * e._z;
    if (a < 0 ? (this._w = -e._w, this._x = -e._x, this._y = -e._y, this._z = -e._z, a = -a) : this.copy(e), a >= 1)
      return this._w = o, this._x = i, this._y = s, this._z = r, this;
    const l = 1 - a * a;
    if (l <= Number.EPSILON) {
      const p = 1 - t;
      return this._w = p * o + t * this._w, this._x = p * i + t * this._x, this._y = p * s + t * this._y, this._z = p * r + t * this._z, this.normalize(), this;
    }
    const c = Math.sqrt(l), u = Math.atan2(c, a), h = Math.sin((1 - t) * u) / c, f = Math.sin(t * u) / c;
    return this._w = o * h + this._w * f, this._x = i * h + this._x * f, this._y = s * h + this._y * f, this._z = r * h + this._z * f, this._onChangeCallback(), this;
  }
  /**
   * Performs a spherical linear interpolation between the given quaternions
   * and stores the result in this quaternion.
   *
   * @param {Quaternion} qa - The source quaternion.
   * @param {Quaternion} qb - The target quaternion.
   * @param {number} t - The interpolation factor in the closed interval `[0, 1]`.
   * @return {Quaternion} A reference to this quaternion.
   */
  slerpQuaternions(e, t, i) {
    return this.copy(e).slerp(t, i);
  }
  /**
   * Sets this quaternion to a uniformly random, normalized quaternion.
   *
   * @return {Quaternion} A reference to this quaternion.
   */
  random() {
    const e = 2 * Math.PI * Math.random(), t = 2 * Math.PI * Math.random(), i = Math.random(), s = Math.sqrt(1 - i), r = Math.sqrt(i);
    return this.set(
      s * Math.sin(e),
      s * Math.cos(e),
      r * Math.sin(t),
      r * Math.cos(t)
    );
  }
  /**
   * Returns `true` if this quaternion is equal with the given one.
   *
   * @param {Quaternion} quaternion - The quaternion to test for equality.
   * @return {boolean} Whether this quaternion is equal with the given one.
   */
  equals(e) {
    return e._x === this._x && e._y === this._y && e._z === this._z && e._w === this._w;
  }
  /**
   * Sets this quaternion's components from the given array.
   *
   * @param {Array<number>} array - An array holding the quaternion component values.
   * @param {number} [offset=0] - The offset into the array.
   * @return {Quaternion} A reference to this quaternion.
   */
  fromArray(e, t = 0) {
    return this._x = e[t], this._y = e[t + 1], this._z = e[t + 2], this._w = e[t + 3], this._onChangeCallback(), this;
  }
  /**
   * Writes the components of this quaternion to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the quaternion components.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The quaternion components.
   */
  toArray(e = [], t = 0) {
    return e[t] = this._x, e[t + 1] = this._y, e[t + 2] = this._z, e[t + 3] = this._w, e;
  }
  /**
   * Sets the components of this quaternion from the given buffer attribute.
   *
   * @param {BufferAttribute} attribute - The buffer attribute holding quaternion data.
   * @param {number} index - The index into the attribute.
   * @return {Quaternion} A reference to this quaternion.
   */
  fromBufferAttribute(e, t) {
    return this._x = e.getX(t), this._y = e.getY(t), this._z = e.getZ(t), this._w = e.getW(t), this._onChangeCallback(), this;
  }
  /**
   * This methods defines the serialization result of this class. Returns the
   * numerical elements of this quaternion in an array of format `[x, y, z, w]`.
   *
   * @return {Array<number>} The serialized quaternion.
   */
  toJSON() {
    return this.toArray();
  }
  _onChange(e) {
    return this._onChangeCallback = e, this;
  }
  _onChangeCallback() {
  }
  *[Symbol.iterator]() {
    yield this._x, yield this._y, yield this._z, yield this._w;
  }
}
class N {
  /**
   * Constructs a new 3D vector.
   *
   * @param {number} [x=0] - The x value of this vector.
   * @param {number} [y=0] - The y value of this vector.
   * @param {number} [z=0] - The z value of this vector.
   */
  constructor(e = 0, t = 0, i = 0) {
    N.prototype.isVector3 = !0, this.x = e, this.y = t, this.z = i;
  }
  /**
   * Sets the vector components.
   *
   * @param {number} x - The value of the x component.
   * @param {number} y - The value of the y component.
   * @param {number} z - The value of the z component.
   * @return {Vector3} A reference to this vector.
   */
  set(e, t, i) {
    return i === void 0 && (i = this.z), this.x = e, this.y = t, this.z = i, this;
  }
  /**
   * Sets the vector components to the same value.
   *
   * @param {number} scalar - The value to set for all vector components.
   * @return {Vector3} A reference to this vector.
   */
  setScalar(e) {
    return this.x = e, this.y = e, this.z = e, this;
  }
  /**
   * Sets the vector's x component to the given value
   *
   * @param {number} x - The value to set.
   * @return {Vector3} A reference to this vector.
   */
  setX(e) {
    return this.x = e, this;
  }
  /**
   * Sets the vector's y component to the given value
   *
   * @param {number} y - The value to set.
   * @return {Vector3} A reference to this vector.
   */
  setY(e) {
    return this.y = e, this;
  }
  /**
   * Sets the vector's z component to the given value
   *
   * @param {number} z - The value to set.
   * @return {Vector3} A reference to this vector.
   */
  setZ(e) {
    return this.z = e, this;
  }
  /**
   * Allows to set a vector component with an index.
   *
   * @param {number} index - The component index. `0` equals to x, `1` equals to y, `2` equals to z.
   * @param {number} value - The value to set.
   * @return {Vector3} A reference to this vector.
   */
  setComponent(e, t) {
    switch (e) {
      case 0:
        this.x = t;
        break;
      case 1:
        this.y = t;
        break;
      case 2:
        this.z = t;
        break;
      default:
        throw new Error("index is out of range: " + e);
    }
    return this;
  }
  /**
   * Returns the value of the vector component which matches the given index.
   *
   * @param {number} index - The component index. `0` equals to x, `1` equals to y, `2` equals to z.
   * @return {number} A vector component value.
   */
  getComponent(e) {
    switch (e) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      case 2:
        return this.z;
      default:
        throw new Error("index is out of range: " + e);
    }
  }
  /**
   * Returns a new vector with copied values from this instance.
   *
   * @return {Vector3} A clone of this instance.
   */
  clone() {
    return new this.constructor(this.x, this.y, this.z);
  }
  /**
   * Copies the values of the given vector to this instance.
   *
   * @param {Vector3} v - The vector to copy.
   * @return {Vector3} A reference to this vector.
   */
  copy(e) {
    return this.x = e.x, this.y = e.y, this.z = e.z, this;
  }
  /**
   * Adds the given vector to this instance.
   *
   * @param {Vector3} v - The vector to add.
   * @return {Vector3} A reference to this vector.
   */
  add(e) {
    return this.x += e.x, this.y += e.y, this.z += e.z, this;
  }
  /**
   * Adds the given scalar value to all components of this instance.
   *
   * @param {number} s - The scalar to add.
   * @return {Vector3} A reference to this vector.
   */
  addScalar(e) {
    return this.x += e, this.y += e, this.z += e, this;
  }
  /**
   * Adds the given vectors and stores the result in this instance.
   *
   * @param {Vector3} a - The first vector.
   * @param {Vector3} b - The second vector.
   * @return {Vector3} A reference to this vector.
   */
  addVectors(e, t) {
    return this.x = e.x + t.x, this.y = e.y + t.y, this.z = e.z + t.z, this;
  }
  /**
   * Adds the given vector scaled by the given factor to this instance.
   *
   * @param {Vector3|Vector4} v - The vector.
   * @param {number} s - The factor that scales `v`.
   * @return {Vector3} A reference to this vector.
   */
  addScaledVector(e, t) {
    return this.x += e.x * t, this.y += e.y * t, this.z += e.z * t, this;
  }
  /**
   * Subtracts the given vector from this instance.
   *
   * @param {Vector3} v - The vector to subtract.
   * @return {Vector3} A reference to this vector.
   */
  sub(e) {
    return this.x -= e.x, this.y -= e.y, this.z -= e.z, this;
  }
  /**
   * Subtracts the given scalar value from all components of this instance.
   *
   * @param {number} s - The scalar to subtract.
   * @return {Vector3} A reference to this vector.
   */
  subScalar(e) {
    return this.x -= e, this.y -= e, this.z -= e, this;
  }
  /**
   * Subtracts the given vectors and stores the result in this instance.
   *
   * @param {Vector3} a - The first vector.
   * @param {Vector3} b - The second vector.
   * @return {Vector3} A reference to this vector.
   */
  subVectors(e, t) {
    return this.x = e.x - t.x, this.y = e.y - t.y, this.z = e.z - t.z, this;
  }
  /**
   * Multiplies the given vector with this instance.
   *
   * @param {Vector3} v - The vector to multiply.
   * @return {Vector3} A reference to this vector.
   */
  multiply(e) {
    return this.x *= e.x, this.y *= e.y, this.z *= e.z, this;
  }
  /**
   * Multiplies the given scalar value with all components of this instance.
   *
   * @param {number} scalar - The scalar to multiply.
   * @return {Vector3} A reference to this vector.
   */
  multiplyScalar(e) {
    return this.x *= e, this.y *= e, this.z *= e, this;
  }
  /**
   * Multiplies the given vectors and stores the result in this instance.
   *
   * @param {Vector3} a - The first vector.
   * @param {Vector3} b - The second vector.
   * @return {Vector3} A reference to this vector.
   */
  multiplyVectors(e, t) {
    return this.x = e.x * t.x, this.y = e.y * t.y, this.z = e.z * t.z, this;
  }
  /**
   * Applies the given Euler rotation to this vector.
   *
   * @param {Euler} euler - The Euler angles.
   * @return {Vector3} A reference to this vector.
   */
  applyEuler(e) {
    return this.applyQuaternion(Iu.setFromEuler(e));
  }
  /**
   * Applies a rotation specified by an axis and an angle to this vector.
   *
   * @param {Vector3} axis - A normalized vector representing the rotation axis.
   * @param {number} angle - The angle in radians.
   * @return {Vector3} A reference to this vector.
   */
  applyAxisAngle(e, t) {
    return this.applyQuaternion(Iu.setFromAxisAngle(e, t));
  }
  /**
   * Multiplies this vector with the given 3x3 matrix.
   *
   * @param {Matrix3} m - The 3x3 matrix.
   * @return {Vector3} A reference to this vector.
   */
  applyMatrix3(e) {
    const t = this.x, i = this.y, s = this.z, r = e.elements;
    return this.x = r[0] * t + r[3] * i + r[6] * s, this.y = r[1] * t + r[4] * i + r[7] * s, this.z = r[2] * t + r[5] * i + r[8] * s, this;
  }
  /**
   * Multiplies this vector by the given normal matrix and normalizes
   * the result.
   *
   * @param {Matrix3} m - The normal matrix.
   * @return {Vector3} A reference to this vector.
   */
  applyNormalMatrix(e) {
    return this.applyMatrix3(e).normalize();
  }
  /**
   * Multiplies this vector (with an implicit 1 in the 4th dimension) by m, and
   * divides by perspective.
   *
   * @param {Matrix4} m - The matrix to apply.
   * @return {Vector3} A reference to this vector.
   */
  applyMatrix4(e) {
    const t = this.x, i = this.y, s = this.z, r = e.elements, o = 1 / (r[3] * t + r[7] * i + r[11] * s + r[15]);
    return this.x = (r[0] * t + r[4] * i + r[8] * s + r[12]) * o, this.y = (r[1] * t + r[5] * i + r[9] * s + r[13]) * o, this.z = (r[2] * t + r[6] * i + r[10] * s + r[14]) * o, this;
  }
  /**
   * Applies the given Quaternion to this vector.
   *
   * @param {Quaternion} q - The Quaternion.
   * @return {Vector3} A reference to this vector.
   */
  applyQuaternion(e) {
    const t = this.x, i = this.y, s = this.z, r = e.x, o = e.y, a = e.z, l = e.w, c = 2 * (o * s - a * i), u = 2 * (a * t - r * s), h = 2 * (r * i - o * t);
    return this.x = t + l * c + o * h - a * u, this.y = i + l * u + a * c - r * h, this.z = s + l * h + r * u - o * c, this;
  }
  /**
   * Projects this vector from world space into the camera's normalized
   * device coordinate (NDC) space.
   *
   * @param {Camera} camera - The camera.
   * @return {Vector3} A reference to this vector.
   */
  project(e) {
    return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix);
  }
  /**
   * Unprojects this vector from the camera's normalized device coordinate (NDC)
   * space into world space.
   *
   * @param {Camera} camera - The camera.
   * @return {Vector3} A reference to this vector.
   */
  unproject(e) {
    return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld);
  }
  /**
   * Transforms the direction of this vector by a matrix (the upper left 3 x 3
   * subset of the given 4x4 matrix and then normalizes the result.
   *
   * @param {Matrix4} m - The matrix.
   * @return {Vector3} A reference to this vector.
   */
  transformDirection(e) {
    const t = this.x, i = this.y, s = this.z, r = e.elements;
    return this.x = r[0] * t + r[4] * i + r[8] * s, this.y = r[1] * t + r[5] * i + r[9] * s, this.z = r[2] * t + r[6] * i + r[10] * s, this.normalize();
  }
  /**
   * Divides this instance by the given vector.
   *
   * @param {Vector3} v - The vector to divide.
   * @return {Vector3} A reference to this vector.
   */
  divide(e) {
    return this.x /= e.x, this.y /= e.y, this.z /= e.z, this;
  }
  /**
   * Divides this vector by the given scalar.
   *
   * @param {number} scalar - The scalar to divide.
   * @return {Vector3} A reference to this vector.
   */
  divideScalar(e) {
    return this.multiplyScalar(1 / e);
  }
  /**
   * If this vector's x, y or z value is greater than the given vector's x, y or z
   * value, replace that value with the corresponding min value.
   *
   * @param {Vector3} v - The vector.
   * @return {Vector3} A reference to this vector.
   */
  min(e) {
    return this.x = Math.min(this.x, e.x), this.y = Math.min(this.y, e.y), this.z = Math.min(this.z, e.z), this;
  }
  /**
   * If this vector's x, y or z value is less than the given vector's x, y or z
   * value, replace that value with the corresponding max value.
   *
   * @param {Vector3} v - The vector.
   * @return {Vector3} A reference to this vector.
   */
  max(e) {
    return this.x = Math.max(this.x, e.x), this.y = Math.max(this.y, e.y), this.z = Math.max(this.z, e.z), this;
  }
  /**
   * If this vector's x, y or z value is greater than the max vector's x, y or z
   * value, it is replaced by the corresponding value.
   * If this vector's x, y or z value is less than the min vector's x, y or z value,
   * it is replaced by the corresponding value.
   *
   * @param {Vector3} min - The minimum x, y and z values.
   * @param {Vector3} max - The maximum x, y and z values in the desired range.
   * @return {Vector3} A reference to this vector.
   */
  clamp(e, t) {
    return this.x = Ke(this.x, e.x, t.x), this.y = Ke(this.y, e.y, t.y), this.z = Ke(this.z, e.z, t.z), this;
  }
  /**
   * If this vector's x, y or z values are greater than the max value, they are
   * replaced by the max value.
   * If this vector's x, y or z values are less than the min value, they are
   * replaced by the min value.
   *
   * @param {number} minVal - The minimum value the components will be clamped to.
   * @param {number} maxVal - The maximum value the components will be clamped to.
   * @return {Vector3} A reference to this vector.
   */
  clampScalar(e, t) {
    return this.x = Ke(this.x, e, t), this.y = Ke(this.y, e, t), this.z = Ke(this.z, e, t), this;
  }
  /**
   * If this vector's length is greater than the max value, it is replaced by
   * the max value.
   * If this vector's length is less than the min value, it is replaced by the
   * min value.
   *
   * @param {number} min - The minimum value the vector length will be clamped to.
   * @param {number} max - The maximum value the vector length will be clamped to.
   * @return {Vector3} A reference to this vector.
   */
  clampLength(e, t) {
    const i = this.length();
    return this.divideScalar(i || 1).multiplyScalar(Ke(i, e, t));
  }
  /**
   * The components of this vector are rounded down to the nearest integer value.
   *
   * @return {Vector3} A reference to this vector.
   */
  floor() {
    return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this.z = Math.floor(this.z), this;
  }
  /**
   * The components of this vector are rounded up to the nearest integer value.
   *
   * @return {Vector3} A reference to this vector.
   */
  ceil() {
    return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this.z = Math.ceil(this.z), this;
  }
  /**
   * The components of this vector are rounded to the nearest integer value
   *
   * @return {Vector3} A reference to this vector.
   */
  round() {
    return this.x = Math.round(this.x), this.y = Math.round(this.y), this.z = Math.round(this.z), this;
  }
  /**
   * The components of this vector are rounded towards zero (up if negative,
   * down if positive) to an integer value.
   *
   * @return {Vector3} A reference to this vector.
   */
  roundToZero() {
    return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this.z = Math.trunc(this.z), this;
  }
  /**
   * Inverts this vector - i.e. sets x = -x, y = -y and z = -z.
   *
   * @return {Vector3} A reference to this vector.
   */
  negate() {
    return this.x = -this.x, this.y = -this.y, this.z = -this.z, this;
  }
  /**
   * Calculates the dot product of the given vector with this instance.
   *
   * @param {Vector3} v - The vector to compute the dot product with.
   * @return {number} The result of the dot product.
   */
  dot(e) {
    return this.x * e.x + this.y * e.y + this.z * e.z;
  }
  // TODO lengthSquared?
  /**
   * Computes the square of the Euclidean length (straight-line length) from
   * (0, 0, 0) to (x, y, z). If you are comparing the lengths of vectors, you should
   * compare the length squared instead as it is slightly more efficient to calculate.
   *
   * @return {number} The square length of this vector.
   */
  lengthSq() {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }
  /**
   * Computes the  Euclidean length (straight-line length) from (0, 0, 0) to (x, y, z).
   *
   * @return {number} The length of this vector.
   */
  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }
  /**
   * Computes the Manhattan length of this vector.
   *
   * @return {number} The length of this vector.
   */
  manhattanLength() {
    return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z);
  }
  /**
   * Converts this vector to a unit vector - that is, sets it equal to a vector
   * with the same direction as this one, but with a vector length of `1`.
   *
   * @return {Vector3} A reference to this vector.
   */
  normalize() {
    return this.divideScalar(this.length() || 1);
  }
  /**
   * Sets this vector to a vector with the same direction as this one, but
   * with the specified length.
   *
   * @param {number} length - The new length of this vector.
   * @return {Vector3} A reference to this vector.
   */
  setLength(e) {
    return this.normalize().multiplyScalar(e);
  }
  /**
   * Linearly interpolates between the given vector and this instance, where
   * alpha is the percent distance along the line - alpha = 0 will be this
   * vector, and alpha = 1 will be the given one.
   *
   * @param {Vector3} v - The vector to interpolate towards.
   * @param {number} alpha - The interpolation factor, typically in the closed interval `[0, 1]`.
   * @return {Vector3} A reference to this vector.
   */
  lerp(e, t) {
    return this.x += (e.x - this.x) * t, this.y += (e.y - this.y) * t, this.z += (e.z - this.z) * t, this;
  }
  /**
   * Linearly interpolates between the given vectors, where alpha is the percent
   * distance along the line - alpha = 0 will be first vector, and alpha = 1 will
   * be the second one. The result is stored in this instance.
   *
   * @param {Vector3} v1 - The first vector.
   * @param {Vector3} v2 - The second vector.
   * @param {number} alpha - The interpolation factor, typically in the closed interval `[0, 1]`.
   * @return {Vector3} A reference to this vector.
   */
  lerpVectors(e, t, i) {
    return this.x = e.x + (t.x - e.x) * i, this.y = e.y + (t.y - e.y) * i, this.z = e.z + (t.z - e.z) * i, this;
  }
  /**
   * Calculates the cross product of the given vector with this instance.
   *
   * @param {Vector3} v - The vector to compute the cross product with.
   * @return {Vector3} The result of the cross product.
   */
  cross(e) {
    return this.crossVectors(this, e);
  }
  /**
   * Calculates the cross product of the given vectors and stores the result
   * in this instance.
   *
   * @param {Vector3} a - The first vector.
   * @param {Vector3} b - The second vector.
   * @return {Vector3} A reference to this vector.
   */
  crossVectors(e, t) {
    const i = e.x, s = e.y, r = e.z, o = t.x, a = t.y, l = t.z;
    return this.x = s * l - r * a, this.y = r * o - i * l, this.z = i * a - s * o, this;
  }
  /**
   * Projects this vector onto the given one.
   *
   * @param {Vector3} v - The vector to project to.
   * @return {Vector3} A reference to this vector.
   */
  projectOnVector(e) {
    const t = e.lengthSq();
    if (t === 0) return this.set(0, 0, 0);
    const i = e.dot(this) / t;
    return this.copy(e).multiplyScalar(i);
  }
  /**
   * Projects this vector onto a plane by subtracting this
   * vector projected onto the plane's normal from this vector.
   *
   * @param {Vector3} planeNormal - The plane normal.
   * @return {Vector3} A reference to this vector.
   */
  projectOnPlane(e) {
    return Sa.copy(this).projectOnVector(e), this.sub(Sa);
  }
  /**
   * Reflects this vector off a plane orthogonal to the given normal vector.
   *
   * @param {Vector3} normal - The (normalized) normal vector.
   * @return {Vector3} A reference to this vector.
   */
  reflect(e) {
    return this.sub(Sa.copy(e).multiplyScalar(2 * this.dot(e)));
  }
  /**
   * Returns the angle between the given vector and this instance in radians.
   *
   * @param {Vector3} v - The vector to compute the angle with.
   * @return {number} The angle in radians.
   */
  angleTo(e) {
    const t = Math.sqrt(this.lengthSq() * e.lengthSq());
    if (t === 0) return Math.PI / 2;
    const i = this.dot(e) / t;
    return Math.acos(Ke(i, -1, 1));
  }
  /**
   * Computes the distance from the given vector to this instance.
   *
   * @param {Vector3} v - The vector to compute the distance to.
   * @return {number} The distance.
   */
  distanceTo(e) {
    return Math.sqrt(this.distanceToSquared(e));
  }
  /**
   * Computes the squared distance from the given vector to this instance.
   * If you are just comparing the distance with another distance, you should compare
   * the distance squared instead as it is slightly more efficient to calculate.
   *
   * @param {Vector3} v - The vector to compute the squared distance to.
   * @return {number} The squared distance.
   */
  distanceToSquared(e) {
    const t = this.x - e.x, i = this.y - e.y, s = this.z - e.z;
    return t * t + i * i + s * s;
  }
  /**
   * Computes the Manhattan distance from the given vector to this instance.
   *
   * @param {Vector3} v - The vector to compute the Manhattan distance to.
   * @return {number} The Manhattan distance.
   */
  manhattanDistanceTo(e) {
    return Math.abs(this.x - e.x) + Math.abs(this.y - e.y) + Math.abs(this.z - e.z);
  }
  /**
   * Sets the vector components from the given spherical coordinates.
   *
   * @param {Spherical} s - The spherical coordinates.
   * @return {Vector3} A reference to this vector.
   */
  setFromSpherical(e) {
    return this.setFromSphericalCoords(e.radius, e.phi, e.theta);
  }
  /**
   * Sets the vector components from the given spherical coordinates.
   *
   * @param {number} radius - The radius.
   * @param {number} phi - The phi angle in radians.
   * @param {number} theta - The theta angle in radians.
   * @return {Vector3} A reference to this vector.
   */
  setFromSphericalCoords(e, t, i) {
    const s = Math.sin(t) * e;
    return this.x = s * Math.sin(i), this.y = Math.cos(t) * e, this.z = s * Math.cos(i), this;
  }
  /**
   * Sets the vector components from the given cylindrical coordinates.
   *
   * @param {Cylindrical} c - The cylindrical coordinates.
   * @return {Vector3} A reference to this vector.
   */
  setFromCylindrical(e) {
    return this.setFromCylindricalCoords(e.radius, e.theta, e.y);
  }
  /**
   * Sets the vector components from the given cylindrical coordinates.
   *
   * @param {number} radius - The radius.
   * @param {number} theta - The theta angle in radians.
   * @param {number} y - The y value.
   * @return {Vector3} A reference to this vector.
   */
  setFromCylindricalCoords(e, t, i) {
    return this.x = e * Math.sin(t), this.y = i, this.z = e * Math.cos(t), this;
  }
  /**
   * Sets the vector components to the position elements of the
   * given transformation matrix.
   *
   * @param {Matrix4} m - The 4x4 matrix.
   * @return {Vector3} A reference to this vector.
   */
  setFromMatrixPosition(e) {
    const t = e.elements;
    return this.x = t[12], this.y = t[13], this.z = t[14], this;
  }
  /**
   * Sets the vector components to the scale elements of the
   * given transformation matrix.
   *
   * @param {Matrix4} m - The 4x4 matrix.
   * @return {Vector3} A reference to this vector.
   */
  setFromMatrixScale(e) {
    const t = this.setFromMatrixColumn(e, 0).length(), i = this.setFromMatrixColumn(e, 1).length(), s = this.setFromMatrixColumn(e, 2).length();
    return this.x = t, this.y = i, this.z = s, this;
  }
  /**
   * Sets the vector components from the specified matrix column.
   *
   * @param {Matrix4} m - The 4x4 matrix.
   * @param {number} index - The column index.
   * @return {Vector3} A reference to this vector.
   */
  setFromMatrixColumn(e, t) {
    return this.fromArray(e.elements, t * 4);
  }
  /**
   * Sets the vector components from the specified matrix column.
   *
   * @param {Matrix3} m - The 3x3 matrix.
   * @param {number} index - The column index.
   * @return {Vector3} A reference to this vector.
   */
  setFromMatrix3Column(e, t) {
    return this.fromArray(e.elements, t * 3);
  }
  /**
   * Sets the vector components from the given Euler angles.
   *
   * @param {Euler} e - The Euler angles to set.
   * @return {Vector3} A reference to this vector.
   */
  setFromEuler(e) {
    return this.x = e._x, this.y = e._y, this.z = e._z, this;
  }
  /**
   * Sets the vector components from the RGB components of the
   * given color.
   *
   * @param {Color} c - The color to set.
   * @return {Vector3} A reference to this vector.
   */
  setFromColor(e) {
    return this.x = e.r, this.y = e.g, this.z = e.b, this;
  }
  /**
   * Returns `true` if this vector is equal with the given one.
   *
   * @param {Vector3} v - The vector to test for equality.
   * @return {boolean} Whether this vector is equal with the given one.
   */
  equals(e) {
    return e.x === this.x && e.y === this.y && e.z === this.z;
  }
  /**
   * Sets this vector's x value to be `array[ offset ]`, y value to be `array[ offset + 1 ]`
   * and z value to be `array[ offset + 2 ]`.
   *
   * @param {Array<number>} array - An array holding the vector component values.
   * @param {number} [offset=0] - The offset into the array.
   * @return {Vector3} A reference to this vector.
   */
  fromArray(e, t = 0) {
    return this.x = e[t], this.y = e[t + 1], this.z = e[t + 2], this;
  }
  /**
   * Writes the components of this vector to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the vector components.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The vector components.
   */
  toArray(e = [], t = 0) {
    return e[t] = this.x, e[t + 1] = this.y, e[t + 2] = this.z, e;
  }
  /**
   * Sets the components of this vector from the given buffer attribute.
   *
   * @param {BufferAttribute} attribute - The buffer attribute holding vector data.
   * @param {number} index - The index into the attribute.
   * @return {Vector3} A reference to this vector.
   */
  fromBufferAttribute(e, t) {
    return this.x = e.getX(t), this.y = e.getY(t), this.z = e.getZ(t), this;
  }
  /**
   * Sets each component of this vector to a pseudo-random value between `0` and
   * `1`, excluding `1`.
   *
   * @return {Vector3} A reference to this vector.
   */
  random() {
    return this.x = Math.random(), this.y = Math.random(), this.z = Math.random(), this;
  }
  /**
   * Sets this vector to a uniformly random point on a unit sphere.
   *
   * @return {Vector3} A reference to this vector.
   */
  randomDirection() {
    const e = Math.random() * Math.PI * 2, t = Math.random() * 2 - 1, i = Math.sqrt(1 - t * t);
    return this.x = i * Math.cos(e), this.y = t, this.z = i * Math.sin(e), this;
  }
  *[Symbol.iterator]() {
    yield this.x, yield this.y, yield this.z;
  }
}
const Sa = /* @__PURE__ */ new N(), Iu = /* @__PURE__ */ new qi();
class qe {
  /**
   * Constructs a new 3x3 matrix. The arguments are supposed to be
   * in row-major order. If no arguments are provided, the constructor
   * initializes the matrix as an identity matrix.
   *
   * @param {number} [n11] - 1-1 matrix element.
   * @param {number} [n12] - 1-2 matrix element.
   * @param {number} [n13] - 1-3 matrix element.
   * @param {number} [n21] - 2-1 matrix element.
   * @param {number} [n22] - 2-2 matrix element.
   * @param {number} [n23] - 2-3 matrix element.
   * @param {number} [n31] - 3-1 matrix element.
   * @param {number} [n32] - 3-2 matrix element.
   * @param {number} [n33] - 3-3 matrix element.
   */
  constructor(e, t, i, s, r, o, a, l, c) {
    qe.prototype.isMatrix3 = !0, this.elements = [
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1
    ], e !== void 0 && this.set(e, t, i, s, r, o, a, l, c);
  }
  /**
   * Sets the elements of the matrix.The arguments are supposed to be
   * in row-major order.
   *
   * @param {number} [n11] - 1-1 matrix element.
   * @param {number} [n12] - 1-2 matrix element.
   * @param {number} [n13] - 1-3 matrix element.
   * @param {number} [n21] - 2-1 matrix element.
   * @param {number} [n22] - 2-2 matrix element.
   * @param {number} [n23] - 2-3 matrix element.
   * @param {number} [n31] - 3-1 matrix element.
   * @param {number} [n32] - 3-2 matrix element.
   * @param {number} [n33] - 3-3 matrix element.
   * @return {Matrix3} A reference to this matrix.
   */
  set(e, t, i, s, r, o, a, l, c) {
    const u = this.elements;
    return u[0] = e, u[1] = s, u[2] = a, u[3] = t, u[4] = r, u[5] = l, u[6] = i, u[7] = o, u[8] = c, this;
  }
  /**
   * Sets this matrix to the 3x3 identity matrix.
   *
   * @return {Matrix3} A reference to this matrix.
   */
  identity() {
    return this.set(
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Copies the values of the given matrix to this instance.
   *
   * @param {Matrix3} m - The matrix to copy.
   * @return {Matrix3} A reference to this matrix.
   */
  copy(e) {
    const t = this.elements, i = e.elements;
    return t[0] = i[0], t[1] = i[1], t[2] = i[2], t[3] = i[3], t[4] = i[4], t[5] = i[5], t[6] = i[6], t[7] = i[7], t[8] = i[8], this;
  }
  /**
   * Extracts the basis of this matrix into the three axis vectors provided.
   *
   * @param {Vector3} xAxis - The basis's x axis.
   * @param {Vector3} yAxis - The basis's y axis.
   * @param {Vector3} zAxis - The basis's z axis.
   * @return {Matrix3} A reference to this matrix.
   */
  extractBasis(e, t, i) {
    return e.setFromMatrix3Column(this, 0), t.setFromMatrix3Column(this, 1), i.setFromMatrix3Column(this, 2), this;
  }
  /**
   * Set this matrix to the upper 3x3 matrix of the given 4x4 matrix.
   *
   * @param {Matrix4} m - The 4x4 matrix.
   * @return {Matrix3} A reference to this matrix.
   */
  setFromMatrix4(e) {
    const t = e.elements;
    return this.set(
      t[0],
      t[4],
      t[8],
      t[1],
      t[5],
      t[9],
      t[2],
      t[6],
      t[10]
    ), this;
  }
  /**
   * Post-multiplies this matrix by the given 3x3 matrix.
   *
   * @param {Matrix3} m - The matrix to multiply with.
   * @return {Matrix3} A reference to this matrix.
   */
  multiply(e) {
    return this.multiplyMatrices(this, e);
  }
  /**
   * Pre-multiplies this matrix by the given 3x3 matrix.
   *
   * @param {Matrix3} m - The matrix to multiply with.
   * @return {Matrix3} A reference to this matrix.
   */
  premultiply(e) {
    return this.multiplyMatrices(e, this);
  }
  /**
   * Multiples the given 3x3 matrices and stores the result
   * in this matrix.
   *
   * @param {Matrix3} a - The first matrix.
   * @param {Matrix3} b - The second matrix.
   * @return {Matrix3} A reference to this matrix.
   */
  multiplyMatrices(e, t) {
    const i = e.elements, s = t.elements, r = this.elements, o = i[0], a = i[3], l = i[6], c = i[1], u = i[4], h = i[7], f = i[2], p = i[5], v = i[8], x = s[0], m = s[3], d = s[6], b = s[1], A = s[4], M = s[7], C = s[2], w = s[5], P = s[8];
    return r[0] = o * x + a * b + l * C, r[3] = o * m + a * A + l * w, r[6] = o * d + a * M + l * P, r[1] = c * x + u * b + h * C, r[4] = c * m + u * A + h * w, r[7] = c * d + u * M + h * P, r[2] = f * x + p * b + v * C, r[5] = f * m + p * A + v * w, r[8] = f * d + p * M + v * P, this;
  }
  /**
   * Multiplies every component of the matrix by the given scalar.
   *
   * @param {number} s - The scalar.
   * @return {Matrix3} A reference to this matrix.
   */
  multiplyScalar(e) {
    const t = this.elements;
    return t[0] *= e, t[3] *= e, t[6] *= e, t[1] *= e, t[4] *= e, t[7] *= e, t[2] *= e, t[5] *= e, t[8] *= e, this;
  }
  /**
   * Computes and returns the determinant of this matrix.
   *
   * @return {number} The determinant.
   */
  determinant() {
    const e = this.elements, t = e[0], i = e[1], s = e[2], r = e[3], o = e[4], a = e[5], l = e[6], c = e[7], u = e[8];
    return t * o * u - t * a * c - i * r * u + i * a * l + s * r * c - s * o * l;
  }
  /**
   * Inverts this matrix, using the [analytic method]{@link https://en.wikipedia.org/wiki/Invertible_matrix#Analytic_solution}.
   * You can not invert with a determinant of zero. If you attempt this, the method produces
   * a zero matrix instead.
   *
   * @return {Matrix3} A reference to this matrix.
   */
  invert() {
    const e = this.elements, t = e[0], i = e[1], s = e[2], r = e[3], o = e[4], a = e[5], l = e[6], c = e[7], u = e[8], h = u * o - a * c, f = a * l - u * r, p = c * r - o * l, v = t * h + i * f + s * p;
    if (v === 0) return this.set(0, 0, 0, 0, 0, 0, 0, 0, 0);
    const x = 1 / v;
    return e[0] = h * x, e[1] = (s * c - u * i) * x, e[2] = (a * i - s * o) * x, e[3] = f * x, e[4] = (u * t - s * l) * x, e[5] = (s * r - a * t) * x, e[6] = p * x, e[7] = (i * l - c * t) * x, e[8] = (o * t - i * r) * x, this;
  }
  /**
   * Transposes this matrix in place.
   *
   * @return {Matrix3} A reference to this matrix.
   */
  transpose() {
    let e;
    const t = this.elements;
    return e = t[1], t[1] = t[3], t[3] = e, e = t[2], t[2] = t[6], t[6] = e, e = t[5], t[5] = t[7], t[7] = e, this;
  }
  /**
   * Computes the normal matrix which is the inverse transpose of the upper
   * left 3x3 portion of the given 4x4 matrix.
   *
   * @param {Matrix4} matrix4 - The 4x4 matrix.
   * @return {Matrix3} A reference to this matrix.
   */
  getNormalMatrix(e) {
    return this.setFromMatrix4(e).invert().transpose();
  }
  /**
   * Transposes this matrix into the supplied array, and returns itself unchanged.
   *
   * @param {Array<number>} r - An array to store the transposed matrix elements.
   * @return {Matrix3} A reference to this matrix.
   */
  transposeIntoArray(e) {
    const t = this.elements;
    return e[0] = t[0], e[1] = t[3], e[2] = t[6], e[3] = t[1], e[4] = t[4], e[5] = t[7], e[6] = t[2], e[7] = t[5], e[8] = t[8], this;
  }
  /**
   * Sets the UV transform matrix from offset, repeat, rotation, and center.
   *
   * @param {number} tx - Offset x.
   * @param {number} ty - Offset y.
   * @param {number} sx - Repeat x.
   * @param {number} sy - Repeat y.
   * @param {number} rotation - Rotation, in radians. Positive values rotate counterclockwise.
   * @param {number} cx - Center x of rotation.
   * @param {number} cy - Center y of rotation
   * @return {Matrix3} A reference to this matrix.
   */
  setUvTransform(e, t, i, s, r, o, a) {
    const l = Math.cos(r), c = Math.sin(r);
    return this.set(
      i * l,
      i * c,
      -i * (l * o + c * a) + o + e,
      -s * c,
      s * l,
      -s * (-c * o + l * a) + a + t,
      0,
      0,
      1
    ), this;
  }
  /**
   * Scales this matrix with the given scalar values.
   *
   * @param {number} sx - The amount to scale in the X axis.
   * @param {number} sy - The amount to scale in the Y axis.
   * @return {Matrix3} A reference to this matrix.
   */
  scale(e, t) {
    return this.premultiply(ya.makeScale(e, t)), this;
  }
  /**
   * Rotates this matrix by the given angle.
   *
   * @param {number} theta - The rotation in radians.
   * @return {Matrix3} A reference to this matrix.
   */
  rotate(e) {
    return this.premultiply(ya.makeRotation(-e)), this;
  }
  /**
   * Translates this matrix by the given scalar values.
   *
   * @param {number} tx - The amount to translate in the X axis.
   * @param {number} ty - The amount to translate in the Y axis.
   * @return {Matrix3} A reference to this matrix.
   */
  translate(e, t) {
    return this.premultiply(ya.makeTranslation(e, t)), this;
  }
  // for 2D Transforms
  /**
   * Sets this matrix as a 2D translation transform.
   *
   * @param {number|Vector2} x - The amount to translate in the X axis or alternatively a translation vector.
   * @param {number} y - The amount to translate in the Y axis.
   * @return {Matrix3} A reference to this matrix.
   */
  makeTranslation(e, t) {
    return e.isVector2 ? this.set(
      1,
      0,
      e.x,
      0,
      1,
      e.y,
      0,
      0,
      1
    ) : this.set(
      1,
      0,
      e,
      0,
      1,
      t,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a 2D rotational transformation.
   *
   * @param {number} theta - The rotation in radians.
   * @return {Matrix3} A reference to this matrix.
   */
  makeRotation(e) {
    const t = Math.cos(e), i = Math.sin(e);
    return this.set(
      t,
      -i,
      0,
      i,
      t,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a 2D scale transform.
   *
   * @param {number} x - The amount to scale in the X axis.
   * @param {number} y - The amount to scale in the Y axis.
   * @return {Matrix3} A reference to this matrix.
   */
  makeScale(e, t) {
    return this.set(
      e,
      0,
      0,
      0,
      t,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Returns `true` if this matrix is equal with the given one.
   *
   * @param {Matrix3} matrix - The matrix to test for equality.
   * @return {boolean} Whether this matrix is equal with the given one.
   */
  equals(e) {
    const t = this.elements, i = e.elements;
    for (let s = 0; s < 9; s++)
      if (t[s] !== i[s]) return !1;
    return !0;
  }
  /**
   * Sets the elements of the matrix from the given array.
   *
   * @param {Array<number>} array - The matrix elements in column-major order.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Matrix3} A reference to this matrix.
   */
  fromArray(e, t = 0) {
    for (let i = 0; i < 9; i++)
      this.elements[i] = e[i + t];
    return this;
  }
  /**
   * Writes the elements of this matrix to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the matrix elements in column-major order.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The matrix elements in column-major order.
   */
  toArray(e = [], t = 0) {
    const i = this.elements;
    return e[t] = i[0], e[t + 1] = i[1], e[t + 2] = i[2], e[t + 3] = i[3], e[t + 4] = i[4], e[t + 5] = i[5], e[t + 6] = i[6], e[t + 7] = i[7], e[t + 8] = i[8], e;
  }
  /**
   * Returns a matrix with copied values from this instance.
   *
   * @return {Matrix3} A clone of this instance.
   */
  clone() {
    return new this.constructor().fromArray(this.elements);
  }
}
const ya = /* @__PURE__ */ new qe();
function pd(n) {
  for (let e = n.length - 1; e >= 0; --e)
    if (n[e] >= 65535) return !0;
  return !1;
}
function Ho(n) {
  return document.createElementNS("http://www.w3.org/1999/xhtml", n);
}
function Ig() {
  const n = Ho("canvas");
  return n.style.display = "block", n;
}
const Uu = {};
function Cr(n) {
  n in Uu || (Uu[n] = !0, console.warn(n));
}
function Ug(n, e, t) {
  return new Promise(function(i, s) {
    function r() {
      switch (n.clientWaitSync(e, n.SYNC_FLUSH_COMMANDS_BIT, 0)) {
        case n.WAIT_FAILED:
          s();
          break;
        case n.TIMEOUT_EXPIRED:
          setTimeout(r, t);
          break;
        default:
          i();
      }
    }
    setTimeout(r, t);
  });
}
const Nu = /* @__PURE__ */ new qe().set(
  0.4123908,
  0.3575843,
  0.1804808,
  0.212639,
  0.7151687,
  0.0721923,
  0.0193308,
  0.1191948,
  0.9505322
), Fu = /* @__PURE__ */ new qe().set(
  3.2409699,
  -1.5373832,
  -0.4986108,
  -0.9692436,
  1.8759675,
  0.0415551,
  0.0556301,
  -0.203977,
  1.0569715
);
function Ng() {
  const n = {
    enabled: !0,
    workingColorSpace: Bs,
    /**
     * Implementations of supported color spaces.
     *
     * Required:
     *	- primaries: chromaticity coordinates [ rx ry gx gy bx by ]
     *	- whitePoint: reference white [ x y ]
     *	- transfer: transfer function (pre-defined)
     *	- toXYZ: Matrix3 RGB to XYZ transform
     *	- fromXYZ: Matrix3 XYZ to RGB transform
     *	- luminanceCoefficients: RGB luminance coefficients
     *
     * Optional:
     *  - outputColorSpaceConfig: { drawingBufferColorSpace: ColorSpace, toneMappingMode: 'extended' | 'standard' }
     *  - workingColorSpaceConfig: { unpackColorSpace: ColorSpace }
     *
     * Reference:
     * - https://www.russellcottrell.com/photo/matrixCalculator.htm
     */
    spaces: {},
    convert: function(s, r, o) {
      return this.enabled === !1 || r === o || !r || !o || (this.spaces[r].transfer === ot && (s.r = ti(s.r), s.g = ti(s.g), s.b = ti(s.b)), this.spaces[r].primaries !== this.spaces[o].primaries && (s.applyMatrix3(this.spaces[r].toXYZ), s.applyMatrix3(this.spaces[o].fromXYZ)), this.spaces[o].transfer === ot && (s.r = Is(s.r), s.g = Is(s.g), s.b = Is(s.b))), s;
    },
    workingToColorSpace: function(s, r) {
      return this.convert(s, this.workingColorSpace, r);
    },
    colorSpaceToWorking: function(s, r) {
      return this.convert(s, r, this.workingColorSpace);
    },
    getPrimaries: function(s) {
      return this.spaces[s].primaries;
    },
    getTransfer: function(s) {
      return s === vi ? Bo : this.spaces[s].transfer;
    },
    getToneMappingMode: function(s) {
      return this.spaces[s].outputColorSpaceConfig.toneMappingMode || "standard";
    },
    getLuminanceCoefficients: function(s, r = this.workingColorSpace) {
      return s.fromArray(this.spaces[r].luminanceCoefficients);
    },
    define: function(s) {
      Object.assign(this.spaces, s);
    },
    // Internal APIs
    _getMatrix: function(s, r, o) {
      return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[o].fromXYZ);
    },
    _getDrawingBufferColorSpace: function(s) {
      return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace;
    },
    _getUnpackColorSpace: function(s = this.workingColorSpace) {
      return this.spaces[s].workingColorSpaceConfig.unpackColorSpace;
    },
    // Deprecated
    fromWorkingColorSpace: function(s, r) {
      return Cr("THREE.ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."), n.workingToColorSpace(s, r);
    },
    toWorkingColorSpace: function(s, r) {
      return Cr("THREE.ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."), n.colorSpaceToWorking(s, r);
    }
  }, e = [0.64, 0.33, 0.3, 0.6, 0.15, 0.06], t = [0.2126, 0.7152, 0.0722], i = [0.3127, 0.329];
  return n.define({
    [Bs]: {
      primaries: e,
      whitePoint: i,
      transfer: Bo,
      toXYZ: Nu,
      fromXYZ: Fu,
      luminanceCoefficients: t,
      workingColorSpaceConfig: { unpackColorSpace: sn },
      outputColorSpaceConfig: { drawingBufferColorSpace: sn }
    },
    [sn]: {
      primaries: e,
      whitePoint: i,
      transfer: ot,
      toXYZ: Nu,
      fromXYZ: Fu,
      luminanceCoefficients: t,
      outputColorSpaceConfig: { drawingBufferColorSpace: sn }
    }
  }), n;
}
const et = /* @__PURE__ */ Ng();
function ti(n) {
  return n < 0.04045 ? n * 0.0773993808 : Math.pow(n * 0.9478672986 + 0.0521327014, 2.4);
}
function Is(n) {
  return n < 31308e-7 ? n * 12.92 : 1.055 * Math.pow(n, 0.41666) - 0.055;
}
let rs;
class Fg {
  /**
   * Returns a data URI containing a representation of the given image.
   *
   * @param {(HTMLImageElement|HTMLCanvasElement)} image - The image object.
   * @param {string} [type='image/png'] - Indicates the image format.
   * @return {string} The data URI.
   */
  static getDataURL(e, t = "image/png") {
    if (/^data:/i.test(e.src) || typeof HTMLCanvasElement > "u")
      return e.src;
    let i;
    if (e instanceof HTMLCanvasElement)
      i = e;
    else {
      rs === void 0 && (rs = Ho("canvas")), rs.width = e.width, rs.height = e.height;
      const s = rs.getContext("2d");
      e instanceof ImageData ? s.putImageData(e, 0, 0) : s.drawImage(e, 0, 0, e.width, e.height), i = rs;
    }
    return i.toDataURL(t);
  }
  /**
   * Converts the given sRGB image data to linear color space.
   *
   * @param {(HTMLImageElement|HTMLCanvasElement|ImageBitmap|Object)} image - The image object.
   * @return {HTMLCanvasElement|Object} The converted image.
   */
  static sRGBToLinear(e) {
    if (typeof HTMLImageElement < "u" && e instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && e instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && e instanceof ImageBitmap) {
      const t = Ho("canvas");
      t.width = e.width, t.height = e.height;
      const i = t.getContext("2d");
      i.drawImage(e, 0, 0, e.width, e.height);
      const s = i.getImageData(0, 0, e.width, e.height), r = s.data;
      for (let o = 0; o < r.length; o++)
        r[o] = ti(r[o] / 255) * 255;
      return i.putImageData(s, 0, 0), t;
    } else if (e.data) {
      const t = e.data.slice(0);
      for (let i = 0; i < t.length; i++)
        t instanceof Uint8Array || t instanceof Uint8ClampedArray ? t[i] = Math.floor(ti(t[i] / 255) * 255) : t[i] = ti(t[i]);
      return {
        data: t,
        width: e.width,
        height: e.height
      };
    } else
      return console.warn("THREE.ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."), e;
  }
}
let Og = 0;
class Ac {
  /**
   * Constructs a new video texture.
   *
   * @param {any} [data=null] - The data definition of a texture.
   */
  constructor(e = null) {
    this.isSource = !0, Object.defineProperty(this, "id", { value: Og++ }), this.uuid = Ur(), this.data = e, this.dataReady = !0, this.version = 0;
  }
  /**
   * Returns the dimensions of the source into the given target vector.
   *
   * @param {(Vector2|Vector3)} target - The target object the result is written into.
   * @return {(Vector2|Vector3)} The dimensions of the source.
   */
  getSize(e) {
    const t = this.data;
    return typeof HTMLVideoElement < "u" && t instanceof HTMLVideoElement ? e.set(t.videoWidth, t.videoHeight, 0) : t instanceof VideoFrame ? e.set(t.displayHeight, t.displayWidth, 0) : t !== null ? e.set(t.width, t.height, t.depth || 0) : e.set(0, 0, 0), e;
  }
  /**
   * When the property is set to `true`, the engine allocates the memory
   * for the texture (if necessary) and triggers the actual texture upload
   * to the GPU next time the source is used.
   *
   * @type {boolean}
   * @default false
   * @param {boolean} value
   */
  set needsUpdate(e) {
    e === !0 && this.version++;
  }
  /**
   * Serializes the source into JSON.
   *
   * @param {?(Object|string)} meta - An optional value holding meta information about the serialization.
   * @return {Object} A JSON object representing the serialized source.
   * @see {@link ObjectLoader#parse}
   */
  toJSON(e) {
    const t = e === void 0 || typeof e == "string";
    if (!t && e.images[this.uuid] !== void 0)
      return e.images[this.uuid];
    const i = {
      uuid: this.uuid,
      url: ""
    }, s = this.data;
    if (s !== null) {
      let r;
      if (Array.isArray(s)) {
        r = [];
        for (let o = 0, a = s.length; o < a; o++)
          s[o].isDataTexture ? r.push(Ea(s[o].image)) : r.push(Ea(s[o]));
      } else
        r = Ea(s);
      i.url = r;
    }
    return t || (e.images[this.uuid] = i), i;
  }
}
function Ea(n) {
  return typeof HTMLImageElement < "u" && n instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && n instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && n instanceof ImageBitmap ? Fg.getDataURL(n) : n.data ? {
    data: Array.from(n.data),
    width: n.width,
    height: n.height,
    type: n.data.constructor.name
  } : (console.warn("THREE.Texture: Unable to serialize Texture."), {});
}
let Bg = 0;
const Ta = /* @__PURE__ */ new N();
class Zt extends Ji {
  /**
   * Constructs a new texture.
   *
   * @param {?Object} [image=Texture.DEFAULT_IMAGE] - The image holding the texture data.
   * @param {number} [mapping=Texture.DEFAULT_MAPPING] - The texture mapping.
   * @param {number} [wrapS=ClampToEdgeWrapping] - The wrapS value.
   * @param {number} [wrapT=ClampToEdgeWrapping] - The wrapT value.
   * @param {number} [magFilter=LinearFilter] - The mag filter value.
   * @param {number} [minFilter=LinearMipmapLinearFilter] - The min filter value.
   * @param {number} [format=RGBAFormat] - The texture format.
   * @param {number} [type=UnsignedByteType] - The texture type.
   * @param {number} [anisotropy=Texture.DEFAULT_ANISOTROPY] - The anisotropy value.
   * @param {string} [colorSpace=NoColorSpace] - The color space.
   */
  constructor(e = Zt.DEFAULT_IMAGE, t = Zt.DEFAULT_MAPPING, i = Vi, s = Vi, r = Un, o = ki, a = xn, l = Bn, c = Zt.DEFAULT_ANISOTROPY, u = vi) {
    super(), this.isTexture = !0, Object.defineProperty(this, "id", { value: Bg++ }), this.uuid = Ur(), this.name = "", this.source = new Ac(e), this.mipmaps = [], this.mapping = t, this.channel = 0, this.wrapS = i, this.wrapT = s, this.magFilter = r, this.minFilter = o, this.anisotropy = c, this.format = a, this.internalFormat = null, this.type = l, this.offset = new Ve(0, 0), this.repeat = new Ve(1, 1), this.center = new Ve(0, 0), this.rotation = 0, this.matrixAutoUpdate = !0, this.matrix = new qe(), this.generateMipmaps = !0, this.premultiplyAlpha = !1, this.flipY = !0, this.unpackAlignment = 4, this.colorSpace = u, this.userData = {}, this.updateRanges = [], this.version = 0, this.onUpdate = null, this.renderTarget = null, this.isRenderTargetTexture = !1, this.isArrayTexture = !!(e && e.depth && e.depth > 1), this.pmremVersion = 0;
  }
  /**
   * The width of the texture in pixels.
   */
  get width() {
    return this.source.getSize(Ta).x;
  }
  /**
   * The height of the texture in pixels.
   */
  get height() {
    return this.source.getSize(Ta).y;
  }
  /**
   * The depth of the texture in pixels.
   */
  get depth() {
    return this.source.getSize(Ta).z;
  }
  /**
   * The image object holding the texture data.
   *
   * @type {?Object}
   */
  get image() {
    return this.source.data;
  }
  set image(e = null) {
    this.source.data = e;
  }
  /**
   * Updates the texture transformation matrix from the from the properties {@link Texture#offset},
   * {@link Texture#repeat}, {@link Texture#rotation}, and {@link Texture#center}.
   */
  updateMatrix() {
    this.matrix.setUvTransform(this.offset.x, this.offset.y, this.repeat.x, this.repeat.y, this.rotation, this.center.x, this.center.y);
  }
  /**
   * Adds a range of data in the data texture to be updated on the GPU.
   *
   * @param {number} start - Position at which to start update.
   * @param {number} count - The number of components to update.
   */
  addUpdateRange(e, t) {
    this.updateRanges.push({ start: e, count: t });
  }
  /**
   * Clears the update ranges.
   */
  clearUpdateRanges() {
    this.updateRanges.length = 0;
  }
  /**
   * Returns a new texture with copied values from this instance.
   *
   * @return {Texture} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Copies the values of the given texture to this instance.
   *
   * @param {Texture} source - The texture to copy.
   * @return {Texture} A reference to this instance.
   */
  copy(e) {
    return this.name = e.name, this.source = e.source, this.mipmaps = e.mipmaps.slice(0), this.mapping = e.mapping, this.channel = e.channel, this.wrapS = e.wrapS, this.wrapT = e.wrapT, this.magFilter = e.magFilter, this.minFilter = e.minFilter, this.anisotropy = e.anisotropy, this.format = e.format, this.internalFormat = e.internalFormat, this.type = e.type, this.offset.copy(e.offset), this.repeat.copy(e.repeat), this.center.copy(e.center), this.rotation = e.rotation, this.matrixAutoUpdate = e.matrixAutoUpdate, this.matrix.copy(e.matrix), this.generateMipmaps = e.generateMipmaps, this.premultiplyAlpha = e.premultiplyAlpha, this.flipY = e.flipY, this.unpackAlignment = e.unpackAlignment, this.colorSpace = e.colorSpace, this.renderTarget = e.renderTarget, this.isRenderTargetTexture = e.isRenderTargetTexture, this.isArrayTexture = e.isArrayTexture, this.userData = JSON.parse(JSON.stringify(e.userData)), this.needsUpdate = !0, this;
  }
  /**
   * Sets this texture's properties based on `values`.
   * @param {Object} values - A container with texture parameters.
   */
  setValues(e) {
    for (const t in e) {
      const i = e[t];
      if (i === void 0) {
        console.warn(`THREE.Texture.setValues(): parameter '${t}' has value of undefined.`);
        continue;
      }
      const s = this[t];
      if (s === void 0) {
        console.warn(`THREE.Texture.setValues(): property '${t}' does not exist.`);
        continue;
      }
      s && i && s.isVector2 && i.isVector2 || s && i && s.isVector3 && i.isVector3 || s && i && s.isMatrix3 && i.isMatrix3 ? s.copy(i) : this[t] = i;
    }
  }
  /**
   * Serializes the texture into JSON.
   *
   * @param {?(Object|string)} meta - An optional value holding meta information about the serialization.
   * @return {Object} A JSON object representing the serialized texture.
   * @see {@link ObjectLoader#parse}
   */
  toJSON(e) {
    const t = e === void 0 || typeof e == "string";
    if (!t && e.textures[this.uuid] !== void 0)
      return e.textures[this.uuid];
    const i = {
      metadata: {
        version: 4.7,
        type: "Texture",
        generator: "Texture.toJSON"
      },
      uuid: this.uuid,
      name: this.name,
      image: this.source.toJSON(e).uuid,
      mapping: this.mapping,
      channel: this.channel,
      repeat: [this.repeat.x, this.repeat.y],
      offset: [this.offset.x, this.offset.y],
      center: [this.center.x, this.center.y],
      rotation: this.rotation,
      wrap: [this.wrapS, this.wrapT],
      format: this.format,
      internalFormat: this.internalFormat,
      type: this.type,
      colorSpace: this.colorSpace,
      minFilter: this.minFilter,
      magFilter: this.magFilter,
      anisotropy: this.anisotropy,
      flipY: this.flipY,
      generateMipmaps: this.generateMipmaps,
      premultiplyAlpha: this.premultiplyAlpha,
      unpackAlignment: this.unpackAlignment
    };
    return Object.keys(this.userData).length > 0 && (i.userData = this.userData), t || (e.textures[this.uuid] = i), i;
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   *
   * @fires Texture#dispose
   */
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
  /**
   * Transforms the given uv vector with the textures uv transformation matrix.
   *
   * @param {Vector2} uv - The uv vector.
   * @return {Vector2} The transformed uv vector.
   */
  transformUv(e) {
    if (this.mapping !== id) return e;
    if (e.applyMatrix3(this.matrix), e.x < 0 || e.x > 1)
      switch (this.wrapS) {
        case El:
          e.x = e.x - Math.floor(e.x);
          break;
        case Vi:
          e.x = e.x < 0 ? 0 : 1;
          break;
        case Tl:
          Math.abs(Math.floor(e.x) % 2) === 1 ? e.x = Math.ceil(e.x) - e.x : e.x = e.x - Math.floor(e.x);
          break;
      }
    if (e.y < 0 || e.y > 1)
      switch (this.wrapT) {
        case El:
          e.y = e.y - Math.floor(e.y);
          break;
        case Vi:
          e.y = e.y < 0 ? 0 : 1;
          break;
        case Tl:
          Math.abs(Math.floor(e.y) % 2) === 1 ? e.y = Math.ceil(e.y) - e.y : e.y = e.y - Math.floor(e.y);
          break;
      }
    return this.flipY && (e.y = 1 - e.y), e;
  }
  /**
   * Setting this property to `true` indicates the engine the texture
   * must be updated in the next render. This triggers a texture upload
   * to the GPU and ensures correct texture parameter configuration.
   *
   * @type {boolean}
   * @default false
   * @param {boolean} value
   */
  set needsUpdate(e) {
    e === !0 && (this.version++, this.source.needsUpdate = !0);
  }
  /**
   * Setting this property to `true` indicates the engine the PMREM
   * must be regenerated.
   *
   * @type {boolean}
   * @default false
   * @param {boolean} value
   */
  set needsPMREMUpdate(e) {
    e === !0 && this.pmremVersion++;
  }
}
Zt.DEFAULT_IMAGE = null;
Zt.DEFAULT_MAPPING = id;
Zt.DEFAULT_ANISOTROPY = 1;
class lt {
  /**
   * Constructs a new 4D vector.
   *
   * @param {number} [x=0] - The x value of this vector.
   * @param {number} [y=0] - The y value of this vector.
   * @param {number} [z=0] - The z value of this vector.
   * @param {number} [w=1] - The w value of this vector.
   */
  constructor(e = 0, t = 0, i = 0, s = 1) {
    lt.prototype.isVector4 = !0, this.x = e, this.y = t, this.z = i, this.w = s;
  }
  /**
   * Alias for {@link Vector4#z}.
   *
   * @type {number}
   */
  get width() {
    return this.z;
  }
  set width(e) {
    this.z = e;
  }
  /**
   * Alias for {@link Vector4#w}.
   *
   * @type {number}
   */
  get height() {
    return this.w;
  }
  set height(e) {
    this.w = e;
  }
  /**
   * Sets the vector components.
   *
   * @param {number} x - The value of the x component.
   * @param {number} y - The value of the y component.
   * @param {number} z - The value of the z component.
   * @param {number} w - The value of the w component.
   * @return {Vector4} A reference to this vector.
   */
  set(e, t, i, s) {
    return this.x = e, this.y = t, this.z = i, this.w = s, this;
  }
  /**
   * Sets the vector components to the same value.
   *
   * @param {number} scalar - The value to set for all vector components.
   * @return {Vector4} A reference to this vector.
   */
  setScalar(e) {
    return this.x = e, this.y = e, this.z = e, this.w = e, this;
  }
  /**
   * Sets the vector's x component to the given value
   *
   * @param {number} x - The value to set.
   * @return {Vector4} A reference to this vector.
   */
  setX(e) {
    return this.x = e, this;
  }
  /**
   * Sets the vector's y component to the given value
   *
   * @param {number} y - The value to set.
   * @return {Vector4} A reference to this vector.
   */
  setY(e) {
    return this.y = e, this;
  }
  /**
   * Sets the vector's z component to the given value
   *
   * @param {number} z - The value to set.
   * @return {Vector4} A reference to this vector.
   */
  setZ(e) {
    return this.z = e, this;
  }
  /**
   * Sets the vector's w component to the given value
   *
   * @param {number} w - The value to set.
   * @return {Vector4} A reference to this vector.
   */
  setW(e) {
    return this.w = e, this;
  }
  /**
   * Allows to set a vector component with an index.
   *
   * @param {number} index - The component index. `0` equals to x, `1` equals to y,
   * `2` equals to z, `3` equals to w.
   * @param {number} value - The value to set.
   * @return {Vector4} A reference to this vector.
   */
  setComponent(e, t) {
    switch (e) {
      case 0:
        this.x = t;
        break;
      case 1:
        this.y = t;
        break;
      case 2:
        this.z = t;
        break;
      case 3:
        this.w = t;
        break;
      default:
        throw new Error("index is out of range: " + e);
    }
    return this;
  }
  /**
   * Returns the value of the vector component which matches the given index.
   *
   * @param {number} index - The component index. `0` equals to x, `1` equals to y,
   * `2` equals to z, `3` equals to w.
   * @return {number} A vector component value.
   */
  getComponent(e) {
    switch (e) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      case 2:
        return this.z;
      case 3:
        return this.w;
      default:
        throw new Error("index is out of range: " + e);
    }
  }
  /**
   * Returns a new vector with copied values from this instance.
   *
   * @return {Vector4} A clone of this instance.
   */
  clone() {
    return new this.constructor(this.x, this.y, this.z, this.w);
  }
  /**
   * Copies the values of the given vector to this instance.
   *
   * @param {Vector3|Vector4} v - The vector to copy.
   * @return {Vector4} A reference to this vector.
   */
  copy(e) {
    return this.x = e.x, this.y = e.y, this.z = e.z, this.w = e.w !== void 0 ? e.w : 1, this;
  }
  /**
   * Adds the given vector to this instance.
   *
   * @param {Vector4} v - The vector to add.
   * @return {Vector4} A reference to this vector.
   */
  add(e) {
    return this.x += e.x, this.y += e.y, this.z += e.z, this.w += e.w, this;
  }
  /**
   * Adds the given scalar value to all components of this instance.
   *
   * @param {number} s - The scalar to add.
   * @return {Vector4} A reference to this vector.
   */
  addScalar(e) {
    return this.x += e, this.y += e, this.z += e, this.w += e, this;
  }
  /**
   * Adds the given vectors and stores the result in this instance.
   *
   * @param {Vector4} a - The first vector.
   * @param {Vector4} b - The second vector.
   * @return {Vector4} A reference to this vector.
   */
  addVectors(e, t) {
    return this.x = e.x + t.x, this.y = e.y + t.y, this.z = e.z + t.z, this.w = e.w + t.w, this;
  }
  /**
   * Adds the given vector scaled by the given factor to this instance.
   *
   * @param {Vector4} v - The vector.
   * @param {number} s - The factor that scales `v`.
   * @return {Vector4} A reference to this vector.
   */
  addScaledVector(e, t) {
    return this.x += e.x * t, this.y += e.y * t, this.z += e.z * t, this.w += e.w * t, this;
  }
  /**
   * Subtracts the given vector from this instance.
   *
   * @param {Vector4} v - The vector to subtract.
   * @return {Vector4} A reference to this vector.
   */
  sub(e) {
    return this.x -= e.x, this.y -= e.y, this.z -= e.z, this.w -= e.w, this;
  }
  /**
   * Subtracts the given scalar value from all components of this instance.
   *
   * @param {number} s - The scalar to subtract.
   * @return {Vector4} A reference to this vector.
   */
  subScalar(e) {
    return this.x -= e, this.y -= e, this.z -= e, this.w -= e, this;
  }
  /**
   * Subtracts the given vectors and stores the result in this instance.
   *
   * @param {Vector4} a - The first vector.
   * @param {Vector4} b - The second vector.
   * @return {Vector4} A reference to this vector.
   */
  subVectors(e, t) {
    return this.x = e.x - t.x, this.y = e.y - t.y, this.z = e.z - t.z, this.w = e.w - t.w, this;
  }
  /**
   * Multiplies the given vector with this instance.
   *
   * @param {Vector4} v - The vector to multiply.
   * @return {Vector4} A reference to this vector.
   */
  multiply(e) {
    return this.x *= e.x, this.y *= e.y, this.z *= e.z, this.w *= e.w, this;
  }
  /**
   * Multiplies the given scalar value with all components of this instance.
   *
   * @param {number} scalar - The scalar to multiply.
   * @return {Vector4} A reference to this vector.
   */
  multiplyScalar(e) {
    return this.x *= e, this.y *= e, this.z *= e, this.w *= e, this;
  }
  /**
   * Multiplies this vector with the given 4x4 matrix.
   *
   * @param {Matrix4} m - The 4x4 matrix.
   * @return {Vector4} A reference to this vector.
   */
  applyMatrix4(e) {
    const t = this.x, i = this.y, s = this.z, r = this.w, o = e.elements;
    return this.x = o[0] * t + o[4] * i + o[8] * s + o[12] * r, this.y = o[1] * t + o[5] * i + o[9] * s + o[13] * r, this.z = o[2] * t + o[6] * i + o[10] * s + o[14] * r, this.w = o[3] * t + o[7] * i + o[11] * s + o[15] * r, this;
  }
  /**
   * Divides this instance by the given vector.
   *
   * @param {Vector4} v - The vector to divide.
   * @return {Vector4} A reference to this vector.
   */
  divide(e) {
    return this.x /= e.x, this.y /= e.y, this.z /= e.z, this.w /= e.w, this;
  }
  /**
   * Divides this vector by the given scalar.
   *
   * @param {number} scalar - The scalar to divide.
   * @return {Vector4} A reference to this vector.
   */
  divideScalar(e) {
    return this.multiplyScalar(1 / e);
  }
  /**
   * Sets the x, y and z components of this
   * vector to the quaternion's axis and w to the angle.
   *
   * @param {Quaternion} q - The Quaternion to set.
   * @return {Vector4} A reference to this vector.
   */
  setAxisAngleFromQuaternion(e) {
    this.w = 2 * Math.acos(e.w);
    const t = Math.sqrt(1 - e.w * e.w);
    return t < 1e-4 ? (this.x = 1, this.y = 0, this.z = 0) : (this.x = e.x / t, this.y = e.y / t, this.z = e.z / t), this;
  }
  /**
   * Sets the x, y and z components of this
   * vector to the axis of rotation and w to the angle.
   *
   * @param {Matrix4} m - A 4x4 matrix of which the upper left 3x3 matrix is a pure rotation matrix.
   * @return {Vector4} A reference to this vector.
   */
  setAxisAngleFromRotationMatrix(e) {
    let t, i, s, r;
    const l = e.elements, c = l[0], u = l[4], h = l[8], f = l[1], p = l[5], v = l[9], x = l[2], m = l[6], d = l[10];
    if (Math.abs(u - f) < 0.01 && Math.abs(h - x) < 0.01 && Math.abs(v - m) < 0.01) {
      if (Math.abs(u + f) < 0.1 && Math.abs(h + x) < 0.1 && Math.abs(v + m) < 0.1 && Math.abs(c + p + d - 3) < 0.1)
        return this.set(1, 0, 0, 0), this;
      t = Math.PI;
      const A = (c + 1) / 2, M = (p + 1) / 2, C = (d + 1) / 2, w = (u + f) / 4, P = (h + x) / 4, U = (v + m) / 4;
      return A > M && A > C ? A < 0.01 ? (i = 0, s = 0.707106781, r = 0.707106781) : (i = Math.sqrt(A), s = w / i, r = P / i) : M > C ? M < 0.01 ? (i = 0.707106781, s = 0, r = 0.707106781) : (s = Math.sqrt(M), i = w / s, r = U / s) : C < 0.01 ? (i = 0.707106781, s = 0.707106781, r = 0) : (r = Math.sqrt(C), i = P / r, s = U / r), this.set(i, s, r, t), this;
    }
    let b = Math.sqrt((m - v) * (m - v) + (h - x) * (h - x) + (f - u) * (f - u));
    return Math.abs(b) < 1e-3 && (b = 1), this.x = (m - v) / b, this.y = (h - x) / b, this.z = (f - u) / b, this.w = Math.acos((c + p + d - 1) / 2), this;
  }
  /**
   * Sets the vector components to the position elements of the
   * given transformation matrix.
   *
   * @param {Matrix4} m - The 4x4 matrix.
   * @return {Vector4} A reference to this vector.
   */
  setFromMatrixPosition(e) {
    const t = e.elements;
    return this.x = t[12], this.y = t[13], this.z = t[14], this.w = t[15], this;
  }
  /**
   * If this vector's x, y, z or w value is greater than the given vector's x, y, z or w
   * value, replace that value with the corresponding min value.
   *
   * @param {Vector4} v - The vector.
   * @return {Vector4} A reference to this vector.
   */
  min(e) {
    return this.x = Math.min(this.x, e.x), this.y = Math.min(this.y, e.y), this.z = Math.min(this.z, e.z), this.w = Math.min(this.w, e.w), this;
  }
  /**
   * If this vector's x, y, z or w value is less than the given vector's x, y, z or w
   * value, replace that value with the corresponding max value.
   *
   * @param {Vector4} v - The vector.
   * @return {Vector4} A reference to this vector.
   */
  max(e) {
    return this.x = Math.max(this.x, e.x), this.y = Math.max(this.y, e.y), this.z = Math.max(this.z, e.z), this.w = Math.max(this.w, e.w), this;
  }
  /**
   * If this vector's x, y, z or w value is greater than the max vector's x, y, z or w
   * value, it is replaced by the corresponding value.
   * If this vector's x, y, z or w value is less than the min vector's x, y, z or w value,
   * it is replaced by the corresponding value.
   *
   * @param {Vector4} min - The minimum x, y and z values.
   * @param {Vector4} max - The maximum x, y and z values in the desired range.
   * @return {Vector4} A reference to this vector.
   */
  clamp(e, t) {
    return this.x = Ke(this.x, e.x, t.x), this.y = Ke(this.y, e.y, t.y), this.z = Ke(this.z, e.z, t.z), this.w = Ke(this.w, e.w, t.w), this;
  }
  /**
   * If this vector's x, y, z or w values are greater than the max value, they are
   * replaced by the max value.
   * If this vector's x, y, z or w values are less than the min value, they are
   * replaced by the min value.
   *
   * @param {number} minVal - The minimum value the components will be clamped to.
   * @param {number} maxVal - The maximum value the components will be clamped to.
   * @return {Vector4} A reference to this vector.
   */
  clampScalar(e, t) {
    return this.x = Ke(this.x, e, t), this.y = Ke(this.y, e, t), this.z = Ke(this.z, e, t), this.w = Ke(this.w, e, t), this;
  }
  /**
   * If this vector's length is greater than the max value, it is replaced by
   * the max value.
   * If this vector's length is less than the min value, it is replaced by the
   * min value.
   *
   * @param {number} min - The minimum value the vector length will be clamped to.
   * @param {number} max - The maximum value the vector length will be clamped to.
   * @return {Vector4} A reference to this vector.
   */
  clampLength(e, t) {
    const i = this.length();
    return this.divideScalar(i || 1).multiplyScalar(Ke(i, e, t));
  }
  /**
   * The components of this vector are rounded down to the nearest integer value.
   *
   * @return {Vector4} A reference to this vector.
   */
  floor() {
    return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this.z = Math.floor(this.z), this.w = Math.floor(this.w), this;
  }
  /**
   * The components of this vector are rounded up to the nearest integer value.
   *
   * @return {Vector4} A reference to this vector.
   */
  ceil() {
    return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this.z = Math.ceil(this.z), this.w = Math.ceil(this.w), this;
  }
  /**
   * The components of this vector are rounded to the nearest integer value
   *
   * @return {Vector4} A reference to this vector.
   */
  round() {
    return this.x = Math.round(this.x), this.y = Math.round(this.y), this.z = Math.round(this.z), this.w = Math.round(this.w), this;
  }
  /**
   * The components of this vector are rounded towards zero (up if negative,
   * down if positive) to an integer value.
   *
   * @return {Vector4} A reference to this vector.
   */
  roundToZero() {
    return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this.z = Math.trunc(this.z), this.w = Math.trunc(this.w), this;
  }
  /**
   * Inverts this vector - i.e. sets x = -x, y = -y, z = -z, w = -w.
   *
   * @return {Vector4} A reference to this vector.
   */
  negate() {
    return this.x = -this.x, this.y = -this.y, this.z = -this.z, this.w = -this.w, this;
  }
  /**
   * Calculates the dot product of the given vector with this instance.
   *
   * @param {Vector4} v - The vector to compute the dot product with.
   * @return {number} The result of the dot product.
   */
  dot(e) {
    return this.x * e.x + this.y * e.y + this.z * e.z + this.w * e.w;
  }
  /**
   * Computes the square of the Euclidean length (straight-line length) from
   * (0, 0, 0, 0) to (x, y, z, w). If you are comparing the lengths of vectors, you should
   * compare the length squared instead as it is slightly more efficient to calculate.
   *
   * @return {number} The square length of this vector.
   */
  lengthSq() {
    return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;
  }
  /**
   * Computes the  Euclidean length (straight-line length) from (0, 0, 0, 0) to (x, y, z, w).
   *
   * @return {number} The length of this vector.
   */
  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w);
  }
  /**
   * Computes the Manhattan length of this vector.
   *
   * @return {number} The length of this vector.
   */
  manhattanLength() {
    return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z) + Math.abs(this.w);
  }
  /**
   * Converts this vector to a unit vector - that is, sets it equal to a vector
   * with the same direction as this one, but with a vector length of `1`.
   *
   * @return {Vector4} A reference to this vector.
   */
  normalize() {
    return this.divideScalar(this.length() || 1);
  }
  /**
   * Sets this vector to a vector with the same direction as this one, but
   * with the specified length.
   *
   * @param {number} length - The new length of this vector.
   * @return {Vector4} A reference to this vector.
   */
  setLength(e) {
    return this.normalize().multiplyScalar(e);
  }
  /**
   * Linearly interpolates between the given vector and this instance, where
   * alpha is the percent distance along the line - alpha = 0 will be this
   * vector, and alpha = 1 will be the given one.
   *
   * @param {Vector4} v - The vector to interpolate towards.
   * @param {number} alpha - The interpolation factor, typically in the closed interval `[0, 1]`.
   * @return {Vector4} A reference to this vector.
   */
  lerp(e, t) {
    return this.x += (e.x - this.x) * t, this.y += (e.y - this.y) * t, this.z += (e.z - this.z) * t, this.w += (e.w - this.w) * t, this;
  }
  /**
   * Linearly interpolates between the given vectors, where alpha is the percent
   * distance along the line - alpha = 0 will be first vector, and alpha = 1 will
   * be the second one. The result is stored in this instance.
   *
   * @param {Vector4} v1 - The first vector.
   * @param {Vector4} v2 - The second vector.
   * @param {number} alpha - The interpolation factor, typically in the closed interval `[0, 1]`.
   * @return {Vector4} A reference to this vector.
   */
  lerpVectors(e, t, i) {
    return this.x = e.x + (t.x - e.x) * i, this.y = e.y + (t.y - e.y) * i, this.z = e.z + (t.z - e.z) * i, this.w = e.w + (t.w - e.w) * i, this;
  }
  /**
   * Returns `true` if this vector is equal with the given one.
   *
   * @param {Vector4} v - The vector to test for equality.
   * @return {boolean} Whether this vector is equal with the given one.
   */
  equals(e) {
    return e.x === this.x && e.y === this.y && e.z === this.z && e.w === this.w;
  }
  /**
   * Sets this vector's x value to be `array[ offset ]`, y value to be `array[ offset + 1 ]`,
   * z value to be `array[ offset + 2 ]`, w value to be `array[ offset + 3 ]`.
   *
   * @param {Array<number>} array - An array holding the vector component values.
   * @param {number} [offset=0] - The offset into the array.
   * @return {Vector4} A reference to this vector.
   */
  fromArray(e, t = 0) {
    return this.x = e[t], this.y = e[t + 1], this.z = e[t + 2], this.w = e[t + 3], this;
  }
  /**
   * Writes the components of this vector to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the vector components.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The vector components.
   */
  toArray(e = [], t = 0) {
    return e[t] = this.x, e[t + 1] = this.y, e[t + 2] = this.z, e[t + 3] = this.w, e;
  }
  /**
   * Sets the components of this vector from the given buffer attribute.
   *
   * @param {BufferAttribute} attribute - The buffer attribute holding vector data.
   * @param {number} index - The index into the attribute.
   * @return {Vector4} A reference to this vector.
   */
  fromBufferAttribute(e, t) {
    return this.x = e.getX(t), this.y = e.getY(t), this.z = e.getZ(t), this.w = e.getW(t), this;
  }
  /**
   * Sets each component of this vector to a pseudo-random value between `0` and
   * `1`, excluding `1`.
   *
   * @return {Vector4} A reference to this vector.
   */
  random() {
    return this.x = Math.random(), this.y = Math.random(), this.z = Math.random(), this.w = Math.random(), this;
  }
  *[Symbol.iterator]() {
    yield this.x, yield this.y, yield this.z, yield this.w;
  }
}
class zg extends Ji {
  /**
   * Render target options.
   *
   * @typedef {Object} RenderTarget~Options
   * @property {boolean} [generateMipmaps=false] - Whether to generate mipmaps or not.
   * @property {number} [magFilter=LinearFilter] - The mag filter.
   * @property {number} [minFilter=LinearFilter] - The min filter.
   * @property {number} [format=RGBAFormat] - The texture format.
   * @property {number} [type=UnsignedByteType] - The texture type.
   * @property {?string} [internalFormat=null] - The texture's internal format.
   * @property {number} [wrapS=ClampToEdgeWrapping] - The texture's uv wrapping mode.
   * @property {number} [wrapT=ClampToEdgeWrapping] - The texture's uv wrapping mode.
   * @property {number} [anisotropy=1] - The texture's anisotropy value.
   * @property {string} [colorSpace=NoColorSpace] - The texture's color space.
   * @property {boolean} [depthBuffer=true] - Whether to allocate a depth buffer or not.
   * @property {boolean} [stencilBuffer=false] - Whether to allocate a stencil buffer or not.
   * @property {boolean} [resolveDepthBuffer=true] - Whether to resolve the depth buffer or not.
   * @property {boolean} [resolveStencilBuffer=true] - Whether  to resolve the stencil buffer or not.
   * @property {?Texture} [depthTexture=null] - Reference to a depth texture.
   * @property {number} [samples=0] - The MSAA samples count.
   * @property {number} [count=1] - Defines the number of color attachments . Must be at least `1`.
   * @property {number} [depth=1] - The texture depth.
   * @property {boolean} [multiview=false] - Whether this target is used for multiview rendering.
   */
  /**
   * Constructs a new render target.
   *
   * @param {number} [width=1] - The width of the render target.
   * @param {number} [height=1] - The height of the render target.
   * @param {RenderTarget~Options} [options] - The configuration object.
   */
  constructor(e = 1, t = 1, i = {}) {
    super(), i = Object.assign({
      generateMipmaps: !1,
      internalFormat: null,
      minFilter: Un,
      depthBuffer: !0,
      stencilBuffer: !1,
      resolveDepthBuffer: !0,
      resolveStencilBuffer: !0,
      depthTexture: null,
      samples: 0,
      count: 1,
      depth: 1,
      multiview: !1
    }, i), this.isRenderTarget = !0, this.width = e, this.height = t, this.depth = i.depth, this.scissor = new lt(0, 0, e, t), this.scissorTest = !1, this.viewport = new lt(0, 0, e, t);
    const s = { width: e, height: t, depth: i.depth }, r = new Zt(s);
    this.textures = [];
    const o = i.count;
    for (let a = 0; a < o; a++)
      this.textures[a] = r.clone(), this.textures[a].isRenderTargetTexture = !0, this.textures[a].renderTarget = this;
    this._setTextureOptions(i), this.depthBuffer = i.depthBuffer, this.stencilBuffer = i.stencilBuffer, this.resolveDepthBuffer = i.resolveDepthBuffer, this.resolveStencilBuffer = i.resolveStencilBuffer, this._depthTexture = null, this.depthTexture = i.depthTexture, this.samples = i.samples, this.multiview = i.multiview;
  }
  _setTextureOptions(e = {}) {
    const t = {
      minFilter: Un,
      generateMipmaps: !1,
      flipY: !1,
      internalFormat: null
    };
    e.mapping !== void 0 && (t.mapping = e.mapping), e.wrapS !== void 0 && (t.wrapS = e.wrapS), e.wrapT !== void 0 && (t.wrapT = e.wrapT), e.wrapR !== void 0 && (t.wrapR = e.wrapR), e.magFilter !== void 0 && (t.magFilter = e.magFilter), e.minFilter !== void 0 && (t.minFilter = e.minFilter), e.format !== void 0 && (t.format = e.format), e.type !== void 0 && (t.type = e.type), e.anisotropy !== void 0 && (t.anisotropy = e.anisotropy), e.colorSpace !== void 0 && (t.colorSpace = e.colorSpace), e.flipY !== void 0 && (t.flipY = e.flipY), e.generateMipmaps !== void 0 && (t.generateMipmaps = e.generateMipmaps), e.internalFormat !== void 0 && (t.internalFormat = e.internalFormat);
    for (let i = 0; i < this.textures.length; i++)
      this.textures[i].setValues(t);
  }
  /**
   * The texture representing the default color attachment.
   *
   * @type {Texture}
   */
  get texture() {
    return this.textures[0];
  }
  set texture(e) {
    this.textures[0] = e;
  }
  set depthTexture(e) {
    this._depthTexture !== null && (this._depthTexture.renderTarget = null), e !== null && (e.renderTarget = this), this._depthTexture = e;
  }
  /**
   * Instead of saving the depth in a renderbuffer, a texture
   * can be used instead which is useful for further processing
   * e.g. in context of post-processing.
   *
   * @type {?DepthTexture}
   * @default null
   */
  get depthTexture() {
    return this._depthTexture;
  }
  /**
   * Sets the size of this render target.
   *
   * @param {number} width - The width.
   * @param {number} height - The height.
   * @param {number} [depth=1] - The depth.
   */
  setSize(e, t, i = 1) {
    if (this.width !== e || this.height !== t || this.depth !== i) {
      this.width = e, this.height = t, this.depth = i;
      for (let s = 0, r = this.textures.length; s < r; s++)
        this.textures[s].image.width = e, this.textures[s].image.height = t, this.textures[s].image.depth = i, this.textures[s].isArrayTexture = this.textures[s].image.depth > 1;
      this.dispose();
    }
    this.viewport.set(0, 0, e, t), this.scissor.set(0, 0, e, t);
  }
  /**
   * Returns a new render target with copied values from this instance.
   *
   * @return {RenderTarget} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Copies the settings of the given render target. This is a structural copy so
   * no resources are shared between render targets after the copy. That includes
   * all MRT textures and the depth texture.
   *
   * @param {RenderTarget} source - The render target to copy.
   * @return {RenderTarget} A reference to this instance.
   */
  copy(e) {
    this.width = e.width, this.height = e.height, this.depth = e.depth, this.scissor.copy(e.scissor), this.scissorTest = e.scissorTest, this.viewport.copy(e.viewport), this.textures.length = 0;
    for (let t = 0, i = e.textures.length; t < i; t++) {
      this.textures[t] = e.textures[t].clone(), this.textures[t].isRenderTargetTexture = !0, this.textures[t].renderTarget = this;
      const s = Object.assign({}, e.textures[t].image);
      this.textures[t].source = new Ac(s);
    }
    return this.depthBuffer = e.depthBuffer, this.stencilBuffer = e.stencilBuffer, this.resolveDepthBuffer = e.resolveDepthBuffer, this.resolveStencilBuffer = e.resolveStencilBuffer, e.depthTexture !== null && (this.depthTexture = e.depthTexture.clone()), this.samples = e.samples, this;
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   *
   * @fires RenderTarget#dispose
   */
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
}
class ji extends zg {
  /**
   * Constructs a new 3D render target.
   *
   * @param {number} [width=1] - The width of the render target.
   * @param {number} [height=1] - The height of the render target.
   * @param {RenderTarget~Options} [options] - The configuration object.
   */
  constructor(e = 1, t = 1, i = {}) {
    super(e, t, i), this.isWebGLRenderTarget = !0;
  }
}
class md extends Zt {
  /**
   * Constructs a new data array texture.
   *
   * @param {?TypedArray} [data=null] - The buffer data.
   * @param {number} [width=1] - The width of the texture.
   * @param {number} [height=1] - The height of the texture.
   * @param {number} [depth=1] - The depth of the texture.
   */
  constructor(e = null, t = 1, i = 1, s = 1) {
    super(null), this.isDataArrayTexture = !0, this.image = { data: e, width: t, height: i, depth: s }, this.magFilter = yn, this.minFilter = yn, this.wrapR = Vi, this.generateMipmaps = !1, this.flipY = !1, this.unpackAlignment = 1, this.layerUpdates = /* @__PURE__ */ new Set();
  }
  /**
   * Describes that a specific layer of the texture needs to be updated.
   * Normally when {@link Texture#needsUpdate} is set to `true`, the
   * entire data texture array is sent to the GPU. Marking specific
   * layers will only transmit subsets of all mipmaps associated with a
   * specific depth in the array which is often much more performant.
   *
   * @param {number} layerIndex - The layer index that should be updated.
   */
  addLayerUpdate(e) {
    this.layerUpdates.add(e);
  }
  /**
   * Resets the layer updates registry.
   */
  clearLayerUpdates() {
    this.layerUpdates.clear();
  }
}
class Hg extends Zt {
  /**
   * Constructs a new data array texture.
   *
   * @param {?TypedArray} [data=null] - The buffer data.
   * @param {number} [width=1] - The width of the texture.
   * @param {number} [height=1] - The height of the texture.
   * @param {number} [depth=1] - The depth of the texture.
   */
  constructor(e = null, t = 1, i = 1, s = 1) {
    super(null), this.isData3DTexture = !0, this.image = { data: e, width: t, height: i, depth: s }, this.magFilter = yn, this.minFilter = yn, this.wrapR = Vi, this.generateMipmaps = !1, this.flipY = !1, this.unpackAlignment = 1;
  }
}
class Nr {
  /**
   * Constructs a new bounding box.
   *
   * @param {Vector3} [min=(Infinity,Infinity,Infinity)] - A vector representing the lower boundary of the box.
   * @param {Vector3} [max=(-Infinity,-Infinity,-Infinity)] - A vector representing the upper boundary of the box.
   */
  constructor(e = new N(1 / 0, 1 / 0, 1 / 0), t = new N(-1 / 0, -1 / 0, -1 / 0)) {
    this.isBox3 = !0, this.min = e, this.max = t;
  }
  /**
   * Sets the lower and upper boundaries of this box.
   * Please note that this method only copies the values from the given objects.
   *
   * @param {Vector3} min - The lower boundary of the box.
   * @param {Vector3} max - The upper boundary of the box.
   * @return {Box3} A reference to this bounding box.
   */
  set(e, t) {
    return this.min.copy(e), this.max.copy(t), this;
  }
  /**
   * Sets the upper and lower bounds of this box so it encloses the position data
   * in the given array.
   *
   * @param {Array<number>} array - An array holding 3D position data.
   * @return {Box3} A reference to this bounding box.
   */
  setFromArray(e) {
    this.makeEmpty();
    for (let t = 0, i = e.length; t < i; t += 3)
      this.expandByPoint(_n.fromArray(e, t));
    return this;
  }
  /**
   * Sets the upper and lower bounds of this box so it encloses the position data
   * in the given buffer attribute.
   *
   * @param {BufferAttribute} attribute - A buffer attribute holding 3D position data.
   * @return {Box3} A reference to this bounding box.
   */
  setFromBufferAttribute(e) {
    this.makeEmpty();
    for (let t = 0, i = e.count; t < i; t++)
      this.expandByPoint(_n.fromBufferAttribute(e, t));
    return this;
  }
  /**
   * Sets the upper and lower bounds of this box so it encloses the position data
   * in the given array.
   *
   * @param {Array<Vector3>} points - An array holding 3D position data as instances of {@link Vector3}.
   * @return {Box3} A reference to this bounding box.
   */
  setFromPoints(e) {
    this.makeEmpty();
    for (let t = 0, i = e.length; t < i; t++)
      this.expandByPoint(e[t]);
    return this;
  }
  /**
   * Centers this box on the given center vector and sets this box's width, height and
   * depth to the given size values.
   *
   * @param {Vector3} center - The center of the box.
   * @param {Vector3} size - The x, y and z dimensions of the box.
   * @return {Box3} A reference to this bounding box.
   */
  setFromCenterAndSize(e, t) {
    const i = _n.copy(t).multiplyScalar(0.5);
    return this.min.copy(e).sub(i), this.max.copy(e).add(i), this;
  }
  /**
   * Computes the world-axis-aligned bounding box for the given 3D object
   * (including its children), accounting for the object's, and children's,
   * world transforms. The function may result in a larger box than strictly necessary.
   *
   * @param {Object3D} object - The 3D object to compute the bounding box for.
   * @param {boolean} [precise=false] - If set to `true`, the method computes the smallest
   * world-axis-aligned bounding box at the expense of more computation.
   * @return {Box3} A reference to this bounding box.
   */
  setFromObject(e, t = !1) {
    return this.makeEmpty(), this.expandByObject(e, t);
  }
  /**
   * Returns a new box with copied values from this instance.
   *
   * @return {Box3} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Copies the values of the given box to this instance.
   *
   * @param {Box3} box - The box to copy.
   * @return {Box3} A reference to this bounding box.
   */
  copy(e) {
    return this.min.copy(e.min), this.max.copy(e.max), this;
  }
  /**
   * Makes this box empty which means in encloses a zero space in 3D.
   *
   * @return {Box3} A reference to this bounding box.
   */
  makeEmpty() {
    return this.min.x = this.min.y = this.min.z = 1 / 0, this.max.x = this.max.y = this.max.z = -1 / 0, this;
  }
  /**
   * Returns true if this box includes zero points within its bounds.
   * Note that a box with equal lower and upper bounds still includes one
   * point, the one both bounds share.
   *
   * @return {boolean} Whether this box is empty or not.
   */
  isEmpty() {
    return this.max.x < this.min.x || this.max.y < this.min.y || this.max.z < this.min.z;
  }
  /**
   * Returns the center point of this box.
   *
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The center point.
   */
  getCenter(e) {
    return this.isEmpty() ? e.set(0, 0, 0) : e.addVectors(this.min, this.max).multiplyScalar(0.5);
  }
  /**
   * Returns the dimensions of this box.
   *
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The size.
   */
  getSize(e) {
    return this.isEmpty() ? e.set(0, 0, 0) : e.subVectors(this.max, this.min);
  }
  /**
   * Expands the boundaries of this box to include the given point.
   *
   * @param {Vector3} point - The point that should be included by the bounding box.
   * @return {Box3} A reference to this bounding box.
   */
  expandByPoint(e) {
    return this.min.min(e), this.max.max(e), this;
  }
  /**
   * Expands this box equilaterally by the given vector. The width of this
   * box will be expanded by the x component of the vector in both
   * directions. The height of this box will be expanded by the y component of
   * the vector in both directions. The depth of this box will be
   * expanded by the z component of the vector in both directions.
   *
   * @param {Vector3} vector - The vector that should expand the bounding box.
   * @return {Box3} A reference to this bounding box.
   */
  expandByVector(e) {
    return this.min.sub(e), this.max.add(e), this;
  }
  /**
   * Expands each dimension of the box by the given scalar. If negative, the
   * dimensions of the box will be contracted.
   *
   * @param {number} scalar - The scalar value that should expand the bounding box.
   * @return {Box3} A reference to this bounding box.
   */
  expandByScalar(e) {
    return this.min.addScalar(-e), this.max.addScalar(e), this;
  }
  /**
   * Expands the boundaries of this box to include the given 3D object and
   * its children, accounting for the object's, and children's, world
   * transforms. The function may result in a larger box than strictly
   * necessary (unless the precise parameter is set to true).
   *
   * @param {Object3D} object - The 3D object that should expand the bounding box.
   * @param {boolean} precise - If set to `true`, the method expands the bounding box
   * as little as necessary at the expense of more computation.
   * @return {Box3} A reference to this bounding box.
   */
  expandByObject(e, t = !1) {
    e.updateWorldMatrix(!1, !1);
    const i = e.geometry;
    if (i !== void 0) {
      const r = i.getAttribute("position");
      if (t === !0 && r !== void 0 && e.isInstancedMesh !== !0)
        for (let o = 0, a = r.count; o < a; o++)
          e.isMesh === !0 ? e.getVertexPosition(o, _n) : _n.fromBufferAttribute(r, o), _n.applyMatrix4(e.matrixWorld), this.expandByPoint(_n);
      else
        e.boundingBox !== void 0 ? (e.boundingBox === null && e.computeBoundingBox(), Gr.copy(e.boundingBox)) : (i.boundingBox === null && i.computeBoundingBox(), Gr.copy(i.boundingBox)), Gr.applyMatrix4(e.matrixWorld), this.union(Gr);
    }
    const s = e.children;
    for (let r = 0, o = s.length; r < o; r++)
      this.expandByObject(s[r], t);
    return this;
  }
  /**
   * Returns `true` if the given point lies within or on the boundaries of this box.
   *
   * @param {Vector3} point - The point to test.
   * @return {boolean} Whether the bounding box contains the given point or not.
   */
  containsPoint(e) {
    return e.x >= this.min.x && e.x <= this.max.x && e.y >= this.min.y && e.y <= this.max.y && e.z >= this.min.z && e.z <= this.max.z;
  }
  /**
   * Returns `true` if this bounding box includes the entirety of the given bounding box.
   * If this box and the given one are identical, this function also returns `true`.
   *
   * @param {Box3} box - The bounding box to test.
   * @return {boolean} Whether the bounding box contains the given bounding box or not.
   */
  containsBox(e) {
    return this.min.x <= e.min.x && e.max.x <= this.max.x && this.min.y <= e.min.y && e.max.y <= this.max.y && this.min.z <= e.min.z && e.max.z <= this.max.z;
  }
  /**
   * Returns a point as a proportion of this box's width, height and depth.
   *
   * @param {Vector3} point - A point in 3D space.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} A point as a proportion of this box's width, height and depth.
   */
  getParameter(e, t) {
    return t.set(
      (e.x - this.min.x) / (this.max.x - this.min.x),
      (e.y - this.min.y) / (this.max.y - this.min.y),
      (e.z - this.min.z) / (this.max.z - this.min.z)
    );
  }
  /**
   * Returns `true` if the given bounding box intersects with this bounding box.
   *
   * @param {Box3} box - The bounding box to test.
   * @return {boolean} Whether the given bounding box intersects with this bounding box.
   */
  intersectsBox(e) {
    return e.max.x >= this.min.x && e.min.x <= this.max.x && e.max.y >= this.min.y && e.min.y <= this.max.y && e.max.z >= this.min.z && e.min.z <= this.max.z;
  }
  /**
   * Returns `true` if the given bounding sphere intersects with this bounding box.
   *
   * @param {Sphere} sphere - The bounding sphere to test.
   * @return {boolean} Whether the given bounding sphere intersects with this bounding box.
   */
  intersectsSphere(e) {
    return this.clampPoint(e.center, _n), _n.distanceToSquared(e.center) <= e.radius * e.radius;
  }
  /**
   * Returns `true` if the given plane intersects with this bounding box.
   *
   * @param {Plane} plane - The plane to test.
   * @return {boolean} Whether the given plane intersects with this bounding box.
   */
  intersectsPlane(e) {
    let t, i;
    return e.normal.x > 0 ? (t = e.normal.x * this.min.x, i = e.normal.x * this.max.x) : (t = e.normal.x * this.max.x, i = e.normal.x * this.min.x), e.normal.y > 0 ? (t += e.normal.y * this.min.y, i += e.normal.y * this.max.y) : (t += e.normal.y * this.max.y, i += e.normal.y * this.min.y), e.normal.z > 0 ? (t += e.normal.z * this.min.z, i += e.normal.z * this.max.z) : (t += e.normal.z * this.max.z, i += e.normal.z * this.min.z), t <= -e.constant && i >= -e.constant;
  }
  /**
   * Returns `true` if the given triangle intersects with this bounding box.
   *
   * @param {Triangle} triangle - The triangle to test.
   * @return {boolean} Whether the given triangle intersects with this bounding box.
   */
  intersectsTriangle(e) {
    if (this.isEmpty())
      return !1;
    this.getCenter($s), Wr.subVectors(this.max, $s), os.subVectors(e.a, $s), as.subVectors(e.b, $s), ls.subVectors(e.c, $s), ai.subVectors(as, os), li.subVectors(ls, as), Pi.subVectors(os, ls);
    let t = [
      0,
      -ai.z,
      ai.y,
      0,
      -li.z,
      li.y,
      0,
      -Pi.z,
      Pi.y,
      ai.z,
      0,
      -ai.x,
      li.z,
      0,
      -li.x,
      Pi.z,
      0,
      -Pi.x,
      -ai.y,
      ai.x,
      0,
      -li.y,
      li.x,
      0,
      -Pi.y,
      Pi.x,
      0
    ];
    return !ba(t, os, as, ls, Wr) || (t = [1, 0, 0, 0, 1, 0, 0, 0, 1], !ba(t, os, as, ls, Wr)) ? !1 : (Xr.crossVectors(ai, li), t = [Xr.x, Xr.y, Xr.z], ba(t, os, as, ls, Wr));
  }
  /**
   * Clamps the given point within the bounds of this box.
   *
   * @param {Vector3} point - The point to clamp.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The clamped point.
   */
  clampPoint(e, t) {
    return t.copy(e).clamp(this.min, this.max);
  }
  /**
   * Returns the euclidean distance from any edge of this box to the specified point. If
   * the given point lies inside of this box, the distance will be `0`.
   *
   * @param {Vector3} point - The point to compute the distance to.
   * @return {number} The euclidean distance.
   */
  distanceToPoint(e) {
    return this.clampPoint(e, _n).distanceTo(e);
  }
  /**
   * Returns a bounding sphere that encloses this bounding box.
   *
   * @param {Sphere} target - The target sphere that is used to store the method's result.
   * @return {Sphere} The bounding sphere that encloses this bounding box.
   */
  getBoundingSphere(e) {
    return this.isEmpty() ? e.makeEmpty() : (this.getCenter(e.center), e.radius = this.getSize(_n).length() * 0.5), e;
  }
  /**
   * Computes the intersection of this bounding box and the given one, setting the upper
   * bound of this box to the lesser of the two boxes' upper bounds and the
   * lower bound of this box to the greater of the two boxes' lower bounds. If
   * there's no overlap, makes this box empty.
   *
   * @param {Box3} box - The bounding box to intersect with.
   * @return {Box3} A reference to this bounding box.
   */
  intersect(e) {
    return this.min.max(e.min), this.max.min(e.max), this.isEmpty() && this.makeEmpty(), this;
  }
  /**
   * Computes the union of this box and another and the given one, setting the upper
   * bound of this box to the greater of the two boxes' upper bounds and the
   * lower bound of this box to the lesser of the two boxes' lower bounds.
   *
   * @param {Box3} box - The bounding box that will be unioned with this instance.
   * @return {Box3} A reference to this bounding box.
   */
  union(e) {
    return this.min.min(e.min), this.max.max(e.max), this;
  }
  /**
   * Transforms this bounding box by the given 4x4 transformation matrix.
   *
   * @param {Matrix4} matrix - The transformation matrix.
   * @return {Box3} A reference to this bounding box.
   */
  applyMatrix4(e) {
    return this.isEmpty() ? this : (Gn[0].set(this.min.x, this.min.y, this.min.z).applyMatrix4(e), Gn[1].set(this.min.x, this.min.y, this.max.z).applyMatrix4(e), Gn[2].set(this.min.x, this.max.y, this.min.z).applyMatrix4(e), Gn[3].set(this.min.x, this.max.y, this.max.z).applyMatrix4(e), Gn[4].set(this.max.x, this.min.y, this.min.z).applyMatrix4(e), Gn[5].set(this.max.x, this.min.y, this.max.z).applyMatrix4(e), Gn[6].set(this.max.x, this.max.y, this.min.z).applyMatrix4(e), Gn[7].set(this.max.x, this.max.y, this.max.z).applyMatrix4(e), this.setFromPoints(Gn), this);
  }
  /**
   * Adds the given offset to both the upper and lower bounds of this bounding box,
   * effectively moving it in 3D space.
   *
   * @param {Vector3} offset - The offset that should be used to translate the bounding box.
   * @return {Box3} A reference to this bounding box.
   */
  translate(e) {
    return this.min.add(e), this.max.add(e), this;
  }
  /**
   * Returns `true` if this bounding box is equal with the given one.
   *
   * @param {Box3} box - The box to test for equality.
   * @return {boolean} Whether this bounding box is equal with the given one.
   */
  equals(e) {
    return e.min.equals(this.min) && e.max.equals(this.max);
  }
  /**
   * Returns a serialized structure of the bounding box.
   *
   * @return {Object} Serialized structure with fields representing the object state.
   */
  toJSON() {
    return {
      min: this.min.toArray(),
      max: this.max.toArray()
    };
  }
  /**
   * Returns a serialized structure of the bounding box.
   *
   * @param {Object} json - The serialized json to set the box from.
   * @return {Box3} A reference to this bounding box.
   */
  fromJSON(e) {
    return this.min.fromArray(e.min), this.max.fromArray(e.max), this;
  }
}
const Gn = [
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N()
], _n = /* @__PURE__ */ new N(), Gr = /* @__PURE__ */ new Nr(), os = /* @__PURE__ */ new N(), as = /* @__PURE__ */ new N(), ls = /* @__PURE__ */ new N(), ai = /* @__PURE__ */ new N(), li = /* @__PURE__ */ new N(), Pi = /* @__PURE__ */ new N(), $s = /* @__PURE__ */ new N(), Wr = /* @__PURE__ */ new N(), Xr = /* @__PURE__ */ new N(), Di = /* @__PURE__ */ new N();
function ba(n, e, t, i, s) {
  for (let r = 0, o = n.length - 3; r <= o; r += 3) {
    Di.fromArray(n, r);
    const a = s.x * Math.abs(Di.x) + s.y * Math.abs(Di.y) + s.z * Math.abs(Di.z), l = e.dot(Di), c = t.dot(Di), u = i.dot(Di);
    if (Math.max(-Math.max(l, c, u), Math.min(l, c, u)) > a)
      return !1;
  }
  return !0;
}
const Vg = /* @__PURE__ */ new Nr(), Zs = /* @__PURE__ */ new N(), Aa = /* @__PURE__ */ new N();
class Fr {
  /**
   * Constructs a new sphere.
   *
   * @param {Vector3} [center=(0,0,0)] - The center of the sphere
   * @param {number} [radius=-1] - The radius of the sphere.
   */
  constructor(e = new N(), t = -1) {
    this.isSphere = !0, this.center = e, this.radius = t;
  }
  /**
   * Sets the sphere's components by copying the given values.
   *
   * @param {Vector3} center - The center.
   * @param {number} radius - The radius.
   * @return {Sphere} A reference to this sphere.
   */
  set(e, t) {
    return this.center.copy(e), this.radius = t, this;
  }
  /**
   * Computes the minimum bounding sphere for list of points.
   * If the optional center point is given, it is used as the sphere's
   * center. Otherwise, the center of the axis-aligned bounding box
   * encompassing the points is calculated.
   *
   * @param {Array<Vector3>} points - A list of points in 3D space.
   * @param {Vector3} [optionalCenter] - The center of the sphere.
   * @return {Sphere} A reference to this sphere.
   */
  setFromPoints(e, t) {
    const i = this.center;
    t !== void 0 ? i.copy(t) : Vg.setFromPoints(e).getCenter(i);
    let s = 0;
    for (let r = 0, o = e.length; r < o; r++)
      s = Math.max(s, i.distanceToSquared(e[r]));
    return this.radius = Math.sqrt(s), this;
  }
  /**
   * Copies the values of the given sphere to this instance.
   *
   * @param {Sphere} sphere - The sphere to copy.
   * @return {Sphere} A reference to this sphere.
   */
  copy(e) {
    return this.center.copy(e.center), this.radius = e.radius, this;
  }
  /**
   * Returns `true` if the sphere is empty (the radius set to a negative number).
   *
   * Spheres with a radius of `0` contain only their center point and are not
   * considered to be empty.
   *
   * @return {boolean} Whether this sphere is empty or not.
   */
  isEmpty() {
    return this.radius < 0;
  }
  /**
   * Makes this sphere empty which means in encloses a zero space in 3D.
   *
   * @return {Sphere} A reference to this sphere.
   */
  makeEmpty() {
    return this.center.set(0, 0, 0), this.radius = -1, this;
  }
  /**
   * Returns `true` if this sphere contains the given point inclusive of
   * the surface of the sphere.
   *
   * @param {Vector3} point - The point to check.
   * @return {boolean} Whether this sphere contains the given point or not.
   */
  containsPoint(e) {
    return e.distanceToSquared(this.center) <= this.radius * this.radius;
  }
  /**
   * Returns the closest distance from the boundary of the sphere to the
   * given point. If the sphere contains the point, the distance will
   * be negative.
   *
   * @param {Vector3} point - The point to compute the distance to.
   * @return {number} The distance to the point.
   */
  distanceToPoint(e) {
    return e.distanceTo(this.center) - this.radius;
  }
  /**
   * Returns `true` if this sphere intersects with the given one.
   *
   * @param {Sphere} sphere - The sphere to test.
   * @return {boolean} Whether this sphere intersects with the given one or not.
   */
  intersectsSphere(e) {
    const t = this.radius + e.radius;
    return e.center.distanceToSquared(this.center) <= t * t;
  }
  /**
   * Returns `true` if this sphere intersects with the given box.
   *
   * @param {Box3} box - The box to test.
   * @return {boolean} Whether this sphere intersects with the given box or not.
   */
  intersectsBox(e) {
    return e.intersectsSphere(this);
  }
  /**
   * Returns `true` if this sphere intersects with the given plane.
   *
   * @param {Plane} plane - The plane to test.
   * @return {boolean} Whether this sphere intersects with the given plane or not.
   */
  intersectsPlane(e) {
    return Math.abs(e.distanceToPoint(this.center)) <= this.radius;
  }
  /**
   * Clamps a point within the sphere. If the point is outside the sphere, it
   * will clamp it to the closest point on the edge of the sphere. Points
   * already inside the sphere will not be affected.
   *
   * @param {Vector3} point - The plane to clamp.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The clamped point.
   */
  clampPoint(e, t) {
    const i = this.center.distanceToSquared(e);
    return t.copy(e), i > this.radius * this.radius && (t.sub(this.center).normalize(), t.multiplyScalar(this.radius).add(this.center)), t;
  }
  /**
   * Returns a bounding box that encloses this sphere.
   *
   * @param {Box3} target - The target box that is used to store the method's result.
   * @return {Box3} The bounding box that encloses this sphere.
   */
  getBoundingBox(e) {
    return this.isEmpty() ? (e.makeEmpty(), e) : (e.set(this.center, this.center), e.expandByScalar(this.radius), e);
  }
  /**
   * Transforms this sphere with the given 4x4 transformation matrix.
   *
   * @param {Matrix4} matrix - The transformation matrix.
   * @return {Sphere} A reference to this sphere.
   */
  applyMatrix4(e) {
    return this.center.applyMatrix4(e), this.radius = this.radius * e.getMaxScaleOnAxis(), this;
  }
  /**
   * Translates the sphere's center by the given offset.
   *
   * @param {Vector3} offset - The offset.
   * @return {Sphere} A reference to this sphere.
   */
  translate(e) {
    return this.center.add(e), this;
  }
  /**
   * Expands the boundaries of this sphere to include the given point.
   *
   * @param {Vector3} point - The point to include.
   * @return {Sphere} A reference to this sphere.
   */
  expandByPoint(e) {
    if (this.isEmpty())
      return this.center.copy(e), this.radius = 0, this;
    Zs.subVectors(e, this.center);
    const t = Zs.lengthSq();
    if (t > this.radius * this.radius) {
      const i = Math.sqrt(t), s = (i - this.radius) * 0.5;
      this.center.addScaledVector(Zs, s / i), this.radius += s;
    }
    return this;
  }
  /**
   * Expands this sphere to enclose both the original sphere and the given sphere.
   *
   * @param {Sphere} sphere - The sphere to include.
   * @return {Sphere} A reference to this sphere.
   */
  union(e) {
    return e.isEmpty() ? this : this.isEmpty() ? (this.copy(e), this) : (this.center.equals(e.center) === !0 ? this.radius = Math.max(this.radius, e.radius) : (Aa.subVectors(e.center, this.center).setLength(e.radius), this.expandByPoint(Zs.copy(e.center).add(Aa)), this.expandByPoint(Zs.copy(e.center).sub(Aa))), this);
  }
  /**
   * Returns `true` if this sphere is equal with the given one.
   *
   * @param {Sphere} sphere - The sphere to test for equality.
   * @return {boolean} Whether this bounding sphere is equal with the given one.
   */
  equals(e) {
    return e.center.equals(this.center) && e.radius === this.radius;
  }
  /**
   * Returns a new sphere with copied values from this instance.
   *
   * @return {Sphere} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Returns a serialized structure of the bounding sphere.
   *
   * @return {Object} Serialized structure with fields representing the object state.
   */
  toJSON() {
    return {
      radius: this.radius,
      center: this.center.toArray()
    };
  }
  /**
   * Returns a serialized structure of the bounding sphere.
   *
   * @param {Object} json - The serialized json to set the sphere from.
   * @return {Box3} A reference to this bounding sphere.
   */
  fromJSON(e) {
    return this.radius = e.radius, this.center.fromArray(e.center), this;
  }
}
const Wn = /* @__PURE__ */ new N(), wa = /* @__PURE__ */ new N(), Yr = /* @__PURE__ */ new N(), ci = /* @__PURE__ */ new N(), Ra = /* @__PURE__ */ new N(), qr = /* @__PURE__ */ new N(), Ca = /* @__PURE__ */ new N();
class na {
  /**
   * Constructs a new ray.
   *
   * @param {Vector3} [origin=(0,0,0)] - The origin of the ray.
   * @param {Vector3} [direction=(0,0,-1)] - The (normalized) direction of the ray.
   */
  constructor(e = new N(), t = new N(0, 0, -1)) {
    this.origin = e, this.direction = t;
  }
  /**
   * Sets the ray's components by copying the given values.
   *
   * @param {Vector3} origin - The origin.
   * @param {Vector3} direction - The direction.
   * @return {Ray} A reference to this ray.
   */
  set(e, t) {
    return this.origin.copy(e), this.direction.copy(t), this;
  }
  /**
   * Copies the values of the given ray to this instance.
   *
   * @param {Ray} ray - The ray to copy.
   * @return {Ray} A reference to this ray.
   */
  copy(e) {
    return this.origin.copy(e.origin), this.direction.copy(e.direction), this;
  }
  /**
   * Returns a vector that is located at a given distance along this ray.
   *
   * @param {number} t - The distance along the ray to retrieve a position for.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} A position on the ray.
   */
  at(e, t) {
    return t.copy(this.origin).addScaledVector(this.direction, e);
  }
  /**
   * Adjusts the direction of the ray to point at the given vector in world space.
   *
   * @param {Vector3} v - The target position.
   * @return {Ray} A reference to this ray.
   */
  lookAt(e) {
    return this.direction.copy(e).sub(this.origin).normalize(), this;
  }
  /**
   * Shift the origin of this ray along its direction by the given distance.
   *
   * @param {number} t - The distance along the ray to interpolate.
   * @return {Ray} A reference to this ray.
   */
  recast(e) {
    return this.origin.copy(this.at(e, Wn)), this;
  }
  /**
   * Returns the point along this ray that is closest to the given point.
   *
   * @param {Vector3} point - A point in 3D space to get the closet location on the ray for.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The closest point on this ray.
   */
  closestPointToPoint(e, t) {
    t.subVectors(e, this.origin);
    const i = t.dot(this.direction);
    return i < 0 ? t.copy(this.origin) : t.copy(this.origin).addScaledVector(this.direction, i);
  }
  /**
   * Returns the distance of the closest approach between this ray and the given point.
   *
   * @param {Vector3} point - A point in 3D space to compute the distance to.
   * @return {number} The distance.
   */
  distanceToPoint(e) {
    return Math.sqrt(this.distanceSqToPoint(e));
  }
  /**
   * Returns the squared distance of the closest approach between this ray and the given point.
   *
   * @param {Vector3} point - A point in 3D space to compute the distance to.
   * @return {number} The squared distance.
   */
  distanceSqToPoint(e) {
    const t = Wn.subVectors(e, this.origin).dot(this.direction);
    return t < 0 ? this.origin.distanceToSquared(e) : (Wn.copy(this.origin).addScaledVector(this.direction, t), Wn.distanceToSquared(e));
  }
  /**
   * Returns the squared distance between this ray and the given line segment.
   *
   * @param {Vector3} v0 - The start point of the line segment.
   * @param {Vector3} v1 - The end point of the line segment.
   * @param {Vector3} [optionalPointOnRay] - When provided, it receives the point on this ray that is closest to the segment.
   * @param {Vector3} [optionalPointOnSegment] - When provided, it receives the point on the line segment that is closest to this ray.
   * @return {number} The squared distance.
   */
  distanceSqToSegment(e, t, i, s) {
    wa.copy(e).add(t).multiplyScalar(0.5), Yr.copy(t).sub(e).normalize(), ci.copy(this.origin).sub(wa);
    const r = e.distanceTo(t) * 0.5, o = -this.direction.dot(Yr), a = ci.dot(this.direction), l = -ci.dot(Yr), c = ci.lengthSq(), u = Math.abs(1 - o * o);
    let h, f, p, v;
    if (u > 0)
      if (h = o * l - a, f = o * a - l, v = r * u, h >= 0)
        if (f >= -v)
          if (f <= v) {
            const x = 1 / u;
            h *= x, f *= x, p = h * (h + o * f + 2 * a) + f * (o * h + f + 2 * l) + c;
          } else
            f = r, h = Math.max(0, -(o * f + a)), p = -h * h + f * (f + 2 * l) + c;
        else
          f = -r, h = Math.max(0, -(o * f + a)), p = -h * h + f * (f + 2 * l) + c;
      else
        f <= -v ? (h = Math.max(0, -(-o * r + a)), f = h > 0 ? -r : Math.min(Math.max(-r, -l), r), p = -h * h + f * (f + 2 * l) + c) : f <= v ? (h = 0, f = Math.min(Math.max(-r, -l), r), p = f * (f + 2 * l) + c) : (h = Math.max(0, -(o * r + a)), f = h > 0 ? r : Math.min(Math.max(-r, -l), r), p = -h * h + f * (f + 2 * l) + c);
    else
      f = o > 0 ? -r : r, h = Math.max(0, -(o * f + a)), p = -h * h + f * (f + 2 * l) + c;
    return i && i.copy(this.origin).addScaledVector(this.direction, h), s && s.copy(wa).addScaledVector(Yr, f), p;
  }
  /**
   * Intersects this ray with the given sphere, returning the intersection
   * point or `null` if there is no intersection.
   *
   * @param {Sphere} sphere - The sphere to intersect.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The intersection point.
   */
  intersectSphere(e, t) {
    Wn.subVectors(e.center, this.origin);
    const i = Wn.dot(this.direction), s = Wn.dot(Wn) - i * i, r = e.radius * e.radius;
    if (s > r) return null;
    const o = Math.sqrt(r - s), a = i - o, l = i + o;
    return l < 0 ? null : a < 0 ? this.at(l, t) : this.at(a, t);
  }
  /**
   * Returns `true` if this ray intersects with the given sphere.
   *
   * @param {Sphere} sphere - The sphere to intersect.
   * @return {boolean} Whether this ray intersects with the given sphere or not.
   */
  intersectsSphere(e) {
    return e.radius < 0 ? !1 : this.distanceSqToPoint(e.center) <= e.radius * e.radius;
  }
  /**
   * Computes the distance from the ray's origin to the given plane. Returns `null` if the ray
   * does not intersect with the plane.
   *
   * @param {Plane} plane - The plane to compute the distance to.
   * @return {?number} Whether this ray intersects with the given sphere or not.
   */
  distanceToPlane(e) {
    const t = e.normal.dot(this.direction);
    if (t === 0)
      return e.distanceToPoint(this.origin) === 0 ? 0 : null;
    const i = -(this.origin.dot(e.normal) + e.constant) / t;
    return i >= 0 ? i : null;
  }
  /**
   * Intersects this ray with the given plane, returning the intersection
   * point or `null` if there is no intersection.
   *
   * @param {Plane} plane - The plane to intersect.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The intersection point.
   */
  intersectPlane(e, t) {
    const i = this.distanceToPlane(e);
    return i === null ? null : this.at(i, t);
  }
  /**
   * Returns `true` if this ray intersects with the given plane.
   *
   * @param {Plane} plane - The plane to intersect.
   * @return {boolean} Whether this ray intersects with the given plane or not.
   */
  intersectsPlane(e) {
    const t = e.distanceToPoint(this.origin);
    return t === 0 || e.normal.dot(this.direction) * t < 0;
  }
  /**
   * Intersects this ray with the given bounding box, returning the intersection
   * point or `null` if there is no intersection.
   *
   * @param {Box3} box - The box to intersect.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The intersection point.
   */
  intersectBox(e, t) {
    let i, s, r, o, a, l;
    const c = 1 / this.direction.x, u = 1 / this.direction.y, h = 1 / this.direction.z, f = this.origin;
    return c >= 0 ? (i = (e.min.x - f.x) * c, s = (e.max.x - f.x) * c) : (i = (e.max.x - f.x) * c, s = (e.min.x - f.x) * c), u >= 0 ? (r = (e.min.y - f.y) * u, o = (e.max.y - f.y) * u) : (r = (e.max.y - f.y) * u, o = (e.min.y - f.y) * u), i > o || r > s || ((r > i || isNaN(i)) && (i = r), (o < s || isNaN(s)) && (s = o), h >= 0 ? (a = (e.min.z - f.z) * h, l = (e.max.z - f.z) * h) : (a = (e.max.z - f.z) * h, l = (e.min.z - f.z) * h), i > l || a > s) || ((a > i || i !== i) && (i = a), (l < s || s !== s) && (s = l), s < 0) ? null : this.at(i >= 0 ? i : s, t);
  }
  /**
   * Returns `true` if this ray intersects with the given box.
   *
   * @param {Box3} box - The box to intersect.
   * @return {boolean} Whether this ray intersects with the given box or not.
   */
  intersectsBox(e) {
    return this.intersectBox(e, Wn) !== null;
  }
  /**
   * Intersects this ray with the given triangle, returning the intersection
   * point or `null` if there is no intersection.
   *
   * @param {Vector3} a - The first vertex of the triangle.
   * @param {Vector3} b - The second vertex of the triangle.
   * @param {Vector3} c - The third vertex of the triangle.
   * @param {boolean} backfaceCulling - Whether to use backface culling or not.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The intersection point.
   */
  intersectTriangle(e, t, i, s, r) {
    Ra.subVectors(t, e), qr.subVectors(i, e), Ca.crossVectors(Ra, qr);
    let o = this.direction.dot(Ca), a;
    if (o > 0) {
      if (s) return null;
      a = 1;
    } else if (o < 0)
      a = -1, o = -o;
    else
      return null;
    ci.subVectors(this.origin, e);
    const l = a * this.direction.dot(qr.crossVectors(ci, qr));
    if (l < 0)
      return null;
    const c = a * this.direction.dot(Ra.cross(ci));
    if (c < 0 || l + c > o)
      return null;
    const u = -a * ci.dot(Ca);
    return u < 0 ? null : this.at(u / o, r);
  }
  /**
   * Transforms this ray with the given 4x4 transformation matrix.
   *
   * @param {Matrix4} matrix4 - The transformation matrix.
   * @return {Ray} A reference to this ray.
   */
  applyMatrix4(e) {
    return this.origin.applyMatrix4(e), this.direction.transformDirection(e), this;
  }
  /**
   * Returns `true` if this ray is equal with the given one.
   *
   * @param {Ray} ray - The ray to test for equality.
   * @return {boolean} Whether this ray is equal with the given one.
   */
  equals(e) {
    return e.origin.equals(this.origin) && e.direction.equals(this.direction);
  }
  /**
   * Returns a new ray with copied values from this instance.
   *
   * @return {Ray} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
}
class pt {
  /**
   * Constructs a new 4x4 matrix. The arguments are supposed to be
   * in row-major order. If no arguments are provided, the constructor
   * initializes the matrix as an identity matrix.
   *
   * @param {number} [n11] - 1-1 matrix element.
   * @param {number} [n12] - 1-2 matrix element.
   * @param {number} [n13] - 1-3 matrix element.
   * @param {number} [n14] - 1-4 matrix element.
   * @param {number} [n21] - 2-1 matrix element.
   * @param {number} [n22] - 2-2 matrix element.
   * @param {number} [n23] - 2-3 matrix element.
   * @param {number} [n24] - 2-4 matrix element.
   * @param {number} [n31] - 3-1 matrix element.
   * @param {number} [n32] - 3-2 matrix element.
   * @param {number} [n33] - 3-3 matrix element.
   * @param {number} [n34] - 3-4 matrix element.
   * @param {number} [n41] - 4-1 matrix element.
   * @param {number} [n42] - 4-2 matrix element.
   * @param {number} [n43] - 4-3 matrix element.
   * @param {number} [n44] - 4-4 matrix element.
   */
  constructor(e, t, i, s, r, o, a, l, c, u, h, f, p, v, x, m) {
    pt.prototype.isMatrix4 = !0, this.elements = [
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ], e !== void 0 && this.set(e, t, i, s, r, o, a, l, c, u, h, f, p, v, x, m);
  }
  /**
   * Sets the elements of the matrix.The arguments are supposed to be
   * in row-major order.
   *
   * @param {number} [n11] - 1-1 matrix element.
   * @param {number} [n12] - 1-2 matrix element.
   * @param {number} [n13] - 1-3 matrix element.
   * @param {number} [n14] - 1-4 matrix element.
   * @param {number} [n21] - 2-1 matrix element.
   * @param {number} [n22] - 2-2 matrix element.
   * @param {number} [n23] - 2-3 matrix element.
   * @param {number} [n24] - 2-4 matrix element.
   * @param {number} [n31] - 3-1 matrix element.
   * @param {number} [n32] - 3-2 matrix element.
   * @param {number} [n33] - 3-3 matrix element.
   * @param {number} [n34] - 3-4 matrix element.
   * @param {number} [n41] - 4-1 matrix element.
   * @param {number} [n42] - 4-2 matrix element.
   * @param {number} [n43] - 4-3 matrix element.
   * @param {number} [n44] - 4-4 matrix element.
   * @return {Matrix4} A reference to this matrix.
   */
  set(e, t, i, s, r, o, a, l, c, u, h, f, p, v, x, m) {
    const d = this.elements;
    return d[0] = e, d[4] = t, d[8] = i, d[12] = s, d[1] = r, d[5] = o, d[9] = a, d[13] = l, d[2] = c, d[6] = u, d[10] = h, d[14] = f, d[3] = p, d[7] = v, d[11] = x, d[15] = m, this;
  }
  /**
   * Sets this matrix to the 4x4 identity matrix.
   *
   * @return {Matrix4} A reference to this matrix.
   */
  identity() {
    return this.set(
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Returns a matrix with copied values from this instance.
   *
   * @return {Matrix4} A clone of this instance.
   */
  clone() {
    return new pt().fromArray(this.elements);
  }
  /**
   * Copies the values of the given matrix to this instance.
   *
   * @param {Matrix4} m - The matrix to copy.
   * @return {Matrix4} A reference to this matrix.
   */
  copy(e) {
    const t = this.elements, i = e.elements;
    return t[0] = i[0], t[1] = i[1], t[2] = i[2], t[3] = i[3], t[4] = i[4], t[5] = i[5], t[6] = i[6], t[7] = i[7], t[8] = i[8], t[9] = i[9], t[10] = i[10], t[11] = i[11], t[12] = i[12], t[13] = i[13], t[14] = i[14], t[15] = i[15], this;
  }
  /**
   * Copies the translation component of the given matrix
   * into this matrix's translation component.
   *
   * @param {Matrix4} m - The matrix to copy the translation component.
   * @return {Matrix4} A reference to this matrix.
   */
  copyPosition(e) {
    const t = this.elements, i = e.elements;
    return t[12] = i[12], t[13] = i[13], t[14] = i[14], this;
  }
  /**
   * Set the upper 3x3 elements of this matrix to the values of given 3x3 matrix.
   *
   * @param {Matrix3} m - The 3x3 matrix.
   * @return {Matrix4} A reference to this matrix.
   */
  setFromMatrix3(e) {
    const t = e.elements;
    return this.set(
      t[0],
      t[3],
      t[6],
      0,
      t[1],
      t[4],
      t[7],
      0,
      t[2],
      t[5],
      t[8],
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Extracts the basis of this matrix into the three axis vectors provided.
   *
   * @param {Vector3} xAxis - The basis's x axis.
   * @param {Vector3} yAxis - The basis's y axis.
   * @param {Vector3} zAxis - The basis's z axis.
   * @return {Matrix4} A reference to this matrix.
   */
  extractBasis(e, t, i) {
    return e.setFromMatrixColumn(this, 0), t.setFromMatrixColumn(this, 1), i.setFromMatrixColumn(this, 2), this;
  }
  /**
   * Sets the given basis vectors to this matrix.
   *
   * @param {Vector3} xAxis - The basis's x axis.
   * @param {Vector3} yAxis - The basis's y axis.
   * @param {Vector3} zAxis - The basis's z axis.
   * @return {Matrix4} A reference to this matrix.
   */
  makeBasis(e, t, i) {
    return this.set(
      e.x,
      t.x,
      i.x,
      0,
      e.y,
      t.y,
      i.y,
      0,
      e.z,
      t.z,
      i.z,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Extracts the rotation component of the given matrix
   * into this matrix's rotation component.
   *
   * Note: This method does not support reflection matrices.
   *
   * @param {Matrix4} m - The matrix.
   * @return {Matrix4} A reference to this matrix.
   */
  extractRotation(e) {
    const t = this.elements, i = e.elements, s = 1 / cs.setFromMatrixColumn(e, 0).length(), r = 1 / cs.setFromMatrixColumn(e, 1).length(), o = 1 / cs.setFromMatrixColumn(e, 2).length();
    return t[0] = i[0] * s, t[1] = i[1] * s, t[2] = i[2] * s, t[3] = 0, t[4] = i[4] * r, t[5] = i[5] * r, t[6] = i[6] * r, t[7] = 0, t[8] = i[8] * o, t[9] = i[9] * o, t[10] = i[10] * o, t[11] = 0, t[12] = 0, t[13] = 0, t[14] = 0, t[15] = 1, this;
  }
  /**
   * Sets the rotation component (the upper left 3x3 matrix) of this matrix to
   * the rotation specified by the given Euler angles. The rest of
   * the matrix is set to the identity. Depending on the {@link Euler#order},
   * there are six possible outcomes. See [this page]{@link https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix}
   * for a complete list.
   *
   * @param {Euler} euler - The Euler angles.
   * @return {Matrix4} A reference to this matrix.
   */
  makeRotationFromEuler(e) {
    const t = this.elements, i = e.x, s = e.y, r = e.z, o = Math.cos(i), a = Math.sin(i), l = Math.cos(s), c = Math.sin(s), u = Math.cos(r), h = Math.sin(r);
    if (e.order === "XYZ") {
      const f = o * u, p = o * h, v = a * u, x = a * h;
      t[0] = l * u, t[4] = -l * h, t[8] = c, t[1] = p + v * c, t[5] = f - x * c, t[9] = -a * l, t[2] = x - f * c, t[6] = v + p * c, t[10] = o * l;
    } else if (e.order === "YXZ") {
      const f = l * u, p = l * h, v = c * u, x = c * h;
      t[0] = f + x * a, t[4] = v * a - p, t[8] = o * c, t[1] = o * h, t[5] = o * u, t[9] = -a, t[2] = p * a - v, t[6] = x + f * a, t[10] = o * l;
    } else if (e.order === "ZXY") {
      const f = l * u, p = l * h, v = c * u, x = c * h;
      t[0] = f - x * a, t[4] = -o * h, t[8] = v + p * a, t[1] = p + v * a, t[5] = o * u, t[9] = x - f * a, t[2] = -o * c, t[6] = a, t[10] = o * l;
    } else if (e.order === "ZYX") {
      const f = o * u, p = o * h, v = a * u, x = a * h;
      t[0] = l * u, t[4] = v * c - p, t[8] = f * c + x, t[1] = l * h, t[5] = x * c + f, t[9] = p * c - v, t[2] = -c, t[6] = a * l, t[10] = o * l;
    } else if (e.order === "YZX") {
      const f = o * l, p = o * c, v = a * l, x = a * c;
      t[0] = l * u, t[4] = x - f * h, t[8] = v * h + p, t[1] = h, t[5] = o * u, t[9] = -a * u, t[2] = -c * u, t[6] = p * h + v, t[10] = f - x * h;
    } else if (e.order === "XZY") {
      const f = o * l, p = o * c, v = a * l, x = a * c;
      t[0] = l * u, t[4] = -h, t[8] = c * u, t[1] = f * h + x, t[5] = o * u, t[9] = p * h - v, t[2] = v * h - p, t[6] = a * u, t[10] = x * h + f;
    }
    return t[3] = 0, t[7] = 0, t[11] = 0, t[12] = 0, t[13] = 0, t[14] = 0, t[15] = 1, this;
  }
  /**
   * Sets the rotation component of this matrix to the rotation specified by
   * the given Quaternion as outlined [here]{@link https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion}
   * The rest of the matrix is set to the identity.
   *
   * @param {Quaternion} q - The Quaternion.
   * @return {Matrix4} A reference to this matrix.
   */
  makeRotationFromQuaternion(e) {
    return this.compose(kg, e, Gg);
  }
  /**
   * Sets the rotation component of the transformation matrix, looking from `eye` towards
   * `target`, and oriented by the up-direction.
   *
   * @param {Vector3} eye - The eye vector.
   * @param {Vector3} target - The target vector.
   * @param {Vector3} up - The up vector.
   * @return {Matrix4} A reference to this matrix.
   */
  lookAt(e, t, i) {
    const s = this.elements;
    return en.subVectors(e, t), en.lengthSq() === 0 && (en.z = 1), en.normalize(), ui.crossVectors(i, en), ui.lengthSq() === 0 && (Math.abs(i.z) === 1 ? en.x += 1e-4 : en.z += 1e-4, en.normalize(), ui.crossVectors(i, en)), ui.normalize(), jr.crossVectors(en, ui), s[0] = ui.x, s[4] = jr.x, s[8] = en.x, s[1] = ui.y, s[5] = jr.y, s[9] = en.y, s[2] = ui.z, s[6] = jr.z, s[10] = en.z, this;
  }
  /**
   * Post-multiplies this matrix by the given 4x4 matrix.
   *
   * @param {Matrix4} m - The matrix to multiply with.
   * @return {Matrix4} A reference to this matrix.
   */
  multiply(e) {
    return this.multiplyMatrices(this, e);
  }
  /**
   * Pre-multiplies this matrix by the given 4x4 matrix.
   *
   * @param {Matrix4} m - The matrix to multiply with.
   * @return {Matrix4} A reference to this matrix.
   */
  premultiply(e) {
    return this.multiplyMatrices(e, this);
  }
  /**
   * Multiples the given 4x4 matrices and stores the result
   * in this matrix.
   *
   * @param {Matrix4} a - The first matrix.
   * @param {Matrix4} b - The second matrix.
   * @return {Matrix4} A reference to this matrix.
   */
  multiplyMatrices(e, t) {
    const i = e.elements, s = t.elements, r = this.elements, o = i[0], a = i[4], l = i[8], c = i[12], u = i[1], h = i[5], f = i[9], p = i[13], v = i[2], x = i[6], m = i[10], d = i[14], b = i[3], A = i[7], M = i[11], C = i[15], w = s[0], P = s[4], U = s[8], S = s[12], y = s[1], D = s[5], L = s[9], V = s[13], Z = s[2], ne = s[6], J = s[10], ie = s[14], H = s[3], fe = s[7], ge = s[11], ye = s[15];
    return r[0] = o * w + a * y + l * Z + c * H, r[4] = o * P + a * D + l * ne + c * fe, r[8] = o * U + a * L + l * J + c * ge, r[12] = o * S + a * V + l * ie + c * ye, r[1] = u * w + h * y + f * Z + p * H, r[5] = u * P + h * D + f * ne + p * fe, r[9] = u * U + h * L + f * J + p * ge, r[13] = u * S + h * V + f * ie + p * ye, r[2] = v * w + x * y + m * Z + d * H, r[6] = v * P + x * D + m * ne + d * fe, r[10] = v * U + x * L + m * J + d * ge, r[14] = v * S + x * V + m * ie + d * ye, r[3] = b * w + A * y + M * Z + C * H, r[7] = b * P + A * D + M * ne + C * fe, r[11] = b * U + A * L + M * J + C * ge, r[15] = b * S + A * V + M * ie + C * ye, this;
  }
  /**
   * Multiplies every component of the matrix by the given scalar.
   *
   * @param {number} s - The scalar.
   * @return {Matrix4} A reference to this matrix.
   */
  multiplyScalar(e) {
    const t = this.elements;
    return t[0] *= e, t[4] *= e, t[8] *= e, t[12] *= e, t[1] *= e, t[5] *= e, t[9] *= e, t[13] *= e, t[2] *= e, t[6] *= e, t[10] *= e, t[14] *= e, t[3] *= e, t[7] *= e, t[11] *= e, t[15] *= e, this;
  }
  /**
   * Computes and returns the determinant of this matrix.
   *
   * Based on the method outlined [here]{@link http://www.euclideanspace.com/maths/algebra/matrix/functions/inverse/fourD/index.html}.
   *
   * @return {number} The determinant.
   */
  determinant() {
    const e = this.elements, t = e[0], i = e[4], s = e[8], r = e[12], o = e[1], a = e[5], l = e[9], c = e[13], u = e[2], h = e[6], f = e[10], p = e[14], v = e[3], x = e[7], m = e[11], d = e[15];
    return v * (+r * l * h - s * c * h - r * a * f + i * c * f + s * a * p - i * l * p) + x * (+t * l * p - t * c * f + r * o * f - s * o * p + s * c * u - r * l * u) + m * (+t * c * h - t * a * p - r * o * h + i * o * p + r * a * u - i * c * u) + d * (-s * a * u - t * l * h + t * a * f + s * o * h - i * o * f + i * l * u);
  }
  /**
   * Transposes this matrix in place.
   *
   * @return {Matrix4} A reference to this matrix.
   */
  transpose() {
    const e = this.elements;
    let t;
    return t = e[1], e[1] = e[4], e[4] = t, t = e[2], e[2] = e[8], e[8] = t, t = e[6], e[6] = e[9], e[9] = t, t = e[3], e[3] = e[12], e[12] = t, t = e[7], e[7] = e[13], e[13] = t, t = e[11], e[11] = e[14], e[14] = t, this;
  }
  /**
   * Sets the position component for this matrix from the given vector,
   * without affecting the rest of the matrix.
   *
   * @param {number|Vector3} x - The x component of the vector or alternatively the vector object.
   * @param {number} y - The y component of the vector.
   * @param {number} z - The z component of the vector.
   * @return {Matrix4} A reference to this matrix.
   */
  setPosition(e, t, i) {
    const s = this.elements;
    return e.isVector3 ? (s[12] = e.x, s[13] = e.y, s[14] = e.z) : (s[12] = e, s[13] = t, s[14] = i), this;
  }
  /**
   * Inverts this matrix, using the [analytic method]{@link https://en.wikipedia.org/wiki/Invertible_matrix#Analytic_solution}.
   * You can not invert with a determinant of zero. If you attempt this, the method produces
   * a zero matrix instead.
   *
   * @return {Matrix4} A reference to this matrix.
   */
  invert() {
    const e = this.elements, t = e[0], i = e[1], s = e[2], r = e[3], o = e[4], a = e[5], l = e[6], c = e[7], u = e[8], h = e[9], f = e[10], p = e[11], v = e[12], x = e[13], m = e[14], d = e[15], b = h * m * c - x * f * c + x * l * p - a * m * p - h * l * d + a * f * d, A = v * f * c - u * m * c - v * l * p + o * m * p + u * l * d - o * f * d, M = u * x * c - v * h * c + v * a * p - o * x * p - u * a * d + o * h * d, C = v * h * l - u * x * l - v * a * f + o * x * f + u * a * m - o * h * m, w = t * b + i * A + s * M + r * C;
    if (w === 0) return this.set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const P = 1 / w;
    return e[0] = b * P, e[1] = (x * f * r - h * m * r - x * s * p + i * m * p + h * s * d - i * f * d) * P, e[2] = (a * m * r - x * l * r + x * s * c - i * m * c - a * s * d + i * l * d) * P, e[3] = (h * l * r - a * f * r - h * s * c + i * f * c + a * s * p - i * l * p) * P, e[4] = A * P, e[5] = (u * m * r - v * f * r + v * s * p - t * m * p - u * s * d + t * f * d) * P, e[6] = (v * l * r - o * m * r - v * s * c + t * m * c + o * s * d - t * l * d) * P, e[7] = (o * f * r - u * l * r + u * s * c - t * f * c - o * s * p + t * l * p) * P, e[8] = M * P, e[9] = (v * h * r - u * x * r - v * i * p + t * x * p + u * i * d - t * h * d) * P, e[10] = (o * x * r - v * a * r + v * i * c - t * x * c - o * i * d + t * a * d) * P, e[11] = (u * a * r - o * h * r - u * i * c + t * h * c + o * i * p - t * a * p) * P, e[12] = C * P, e[13] = (u * x * s - v * h * s + v * i * f - t * x * f - u * i * m + t * h * m) * P, e[14] = (v * a * s - o * x * s - v * i * l + t * x * l + o * i * m - t * a * m) * P, e[15] = (o * h * s - u * a * s + u * i * l - t * h * l - o * i * f + t * a * f) * P, this;
  }
  /**
   * Multiplies the columns of this matrix by the given vector.
   *
   * @param {Vector3} v - The scale vector.
   * @return {Matrix4} A reference to this matrix.
   */
  scale(e) {
    const t = this.elements, i = e.x, s = e.y, r = e.z;
    return t[0] *= i, t[4] *= s, t[8] *= r, t[1] *= i, t[5] *= s, t[9] *= r, t[2] *= i, t[6] *= s, t[10] *= r, t[3] *= i, t[7] *= s, t[11] *= r, this;
  }
  /**
   * Gets the maximum scale value of the three axes.
   *
   * @return {number} The maximum scale.
   */
  getMaxScaleOnAxis() {
    const e = this.elements, t = e[0] * e[0] + e[1] * e[1] + e[2] * e[2], i = e[4] * e[4] + e[5] * e[5] + e[6] * e[6], s = e[8] * e[8] + e[9] * e[9] + e[10] * e[10];
    return Math.sqrt(Math.max(t, i, s));
  }
  /**
   * Sets this matrix as a translation transform from the given vector.
   *
   * @param {number|Vector3} x - The amount to translate in the X axis or alternatively a translation vector.
   * @param {number} y - The amount to translate in the Y axis.
   * @param {number} z - The amount to translate in the z axis.
   * @return {Matrix4} A reference to this matrix.
   */
  makeTranslation(e, t, i) {
    return e.isVector3 ? this.set(
      1,
      0,
      0,
      e.x,
      0,
      1,
      0,
      e.y,
      0,
      0,
      1,
      e.z,
      0,
      0,
      0,
      1
    ) : this.set(
      1,
      0,
      0,
      e,
      0,
      1,
      0,
      t,
      0,
      0,
      1,
      i,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a rotational transformation around the X axis by
   * the given angle.
   *
   * @param {number} theta - The rotation in radians.
   * @return {Matrix4} A reference to this matrix.
   */
  makeRotationX(e) {
    const t = Math.cos(e), i = Math.sin(e);
    return this.set(
      1,
      0,
      0,
      0,
      0,
      t,
      -i,
      0,
      0,
      i,
      t,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a rotational transformation around the Y axis by
   * the given angle.
   *
   * @param {number} theta - The rotation in radians.
   * @return {Matrix4} A reference to this matrix.
   */
  makeRotationY(e) {
    const t = Math.cos(e), i = Math.sin(e);
    return this.set(
      t,
      0,
      i,
      0,
      0,
      1,
      0,
      0,
      -i,
      0,
      t,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a rotational transformation around the Z axis by
   * the given angle.
   *
   * @param {number} theta - The rotation in radians.
   * @return {Matrix4} A reference to this matrix.
   */
  makeRotationZ(e) {
    const t = Math.cos(e), i = Math.sin(e);
    return this.set(
      t,
      -i,
      0,
      0,
      i,
      t,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a rotational transformation around the given axis by
   * the given angle.
   *
   * This is a somewhat controversial but mathematically sound alternative to
   * rotating via Quaternions. See the discussion [here]{@link https://www.gamedev.net/articles/programming/math-and-physics/do-we-really-need-quaternions-r1199}.
   *
   * @param {Vector3} axis - The normalized rotation axis.
   * @param {number} angle - The rotation in radians.
   * @return {Matrix4} A reference to this matrix.
   */
  makeRotationAxis(e, t) {
    const i = Math.cos(t), s = Math.sin(t), r = 1 - i, o = e.x, a = e.y, l = e.z, c = r * o, u = r * a;
    return this.set(
      c * o + i,
      c * a - s * l,
      c * l + s * a,
      0,
      c * a + s * l,
      u * a + i,
      u * l - s * o,
      0,
      c * l - s * a,
      u * l + s * o,
      r * l * l + i,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a scale transformation.
   *
   * @param {number} x - The amount to scale in the X axis.
   * @param {number} y - The amount to scale in the Y axis.
   * @param {number} z - The amount to scale in the Z axis.
   * @return {Matrix4} A reference to this matrix.
   */
  makeScale(e, t, i) {
    return this.set(
      e,
      0,
      0,
      0,
      0,
      t,
      0,
      0,
      0,
      0,
      i,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a shear transformation.
   *
   * @param {number} xy - The amount to shear X by Y.
   * @param {number} xz - The amount to shear X by Z.
   * @param {number} yx - The amount to shear Y by X.
   * @param {number} yz - The amount to shear Y by Z.
   * @param {number} zx - The amount to shear Z by X.
   * @param {number} zy - The amount to shear Z by Y.
   * @return {Matrix4} A reference to this matrix.
   */
  makeShear(e, t, i, s, r, o) {
    return this.set(
      1,
      i,
      r,
      0,
      e,
      1,
      o,
      0,
      t,
      s,
      1,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix to the transformation composed of the given position,
   * rotation (Quaternion) and scale.
   *
   * @param {Vector3} position - The position vector.
   * @param {Quaternion} quaternion - The rotation as a Quaternion.
   * @param {Vector3} scale - The scale vector.
   * @return {Matrix4} A reference to this matrix.
   */
  compose(e, t, i) {
    const s = this.elements, r = t._x, o = t._y, a = t._z, l = t._w, c = r + r, u = o + o, h = a + a, f = r * c, p = r * u, v = r * h, x = o * u, m = o * h, d = a * h, b = l * c, A = l * u, M = l * h, C = i.x, w = i.y, P = i.z;
    return s[0] = (1 - (x + d)) * C, s[1] = (p + M) * C, s[2] = (v - A) * C, s[3] = 0, s[4] = (p - M) * w, s[5] = (1 - (f + d)) * w, s[6] = (m + b) * w, s[7] = 0, s[8] = (v + A) * P, s[9] = (m - b) * P, s[10] = (1 - (f + x)) * P, s[11] = 0, s[12] = e.x, s[13] = e.y, s[14] = e.z, s[15] = 1, this;
  }
  /**
   * Decomposes this matrix into its position, rotation and scale components
   * and provides the result in the given objects.
   *
   * Note: Not all matrices are decomposable in this way. For example, if an
   * object has a non-uniformly scaled parent, then the object's world matrix
   * may not be decomposable, and this method may not be appropriate.
   *
   * @param {Vector3} position - The position vector.
   * @param {Quaternion} quaternion - The rotation as a Quaternion.
   * @param {Vector3} scale - The scale vector.
   * @return {Matrix4} A reference to this matrix.
   */
  decompose(e, t, i) {
    const s = this.elements;
    let r = cs.set(s[0], s[1], s[2]).length();
    const o = cs.set(s[4], s[5], s[6]).length(), a = cs.set(s[8], s[9], s[10]).length();
    this.determinant() < 0 && (r = -r), e.x = s[12], e.y = s[13], e.z = s[14], gn.copy(this);
    const c = 1 / r, u = 1 / o, h = 1 / a;
    return gn.elements[0] *= c, gn.elements[1] *= c, gn.elements[2] *= c, gn.elements[4] *= u, gn.elements[5] *= u, gn.elements[6] *= u, gn.elements[8] *= h, gn.elements[9] *= h, gn.elements[10] *= h, t.setFromRotationMatrix(gn), i.x = r, i.y = o, i.z = a, this;
  }
  /**
	 * Creates a perspective projection matrix. This is used internally by
	 * {@link PerspectiveCamera#updateProjectionMatrix}.

	 * @param {number} left - Left boundary of the viewing frustum at the near plane.
	 * @param {number} right - Right boundary of the viewing frustum at the near plane.
	 * @param {number} top - Top boundary of the viewing frustum at the near plane.
	 * @param {number} bottom - Bottom boundary of the viewing frustum at the near plane.
	 * @param {number} near - The distance from the camera to the near plane.
	 * @param {number} far - The distance from the camera to the far plane.
	 * @param {(WebGLCoordinateSystem|WebGPUCoordinateSystem)} [coordinateSystem=WebGLCoordinateSystem] - The coordinate system.
	 * @param {boolean} [reversedDepth=false] - Whether to use a reversed depth.
	 * @return {Matrix4} A reference to this matrix.
	 */
  makePerspective(e, t, i, s, r, o, a = Nn, l = !1) {
    const c = this.elements, u = 2 * r / (t - e), h = 2 * r / (i - s), f = (t + e) / (t - e), p = (i + s) / (i - s);
    let v, x;
    if (l)
      v = r / (o - r), x = o * r / (o - r);
    else if (a === Nn)
      v = -(o + r) / (o - r), x = -2 * o * r / (o - r);
    else if (a === zo)
      v = -o / (o - r), x = -o * r / (o - r);
    else
      throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: " + a);
    return c[0] = u, c[4] = 0, c[8] = f, c[12] = 0, c[1] = 0, c[5] = h, c[9] = p, c[13] = 0, c[2] = 0, c[6] = 0, c[10] = v, c[14] = x, c[3] = 0, c[7] = 0, c[11] = -1, c[15] = 0, this;
  }
  /**
	 * Creates a orthographic projection matrix. This is used internally by
	 * {@link OrthographicCamera#updateProjectionMatrix}.

	 * @param {number} left - Left boundary of the viewing frustum at the near plane.
	 * @param {number} right - Right boundary of the viewing frustum at the near plane.
	 * @param {number} top - Top boundary of the viewing frustum at the near plane.
	 * @param {number} bottom - Bottom boundary of the viewing frustum at the near plane.
	 * @param {number} near - The distance from the camera to the near plane.
	 * @param {number} far - The distance from the camera to the far plane.
	 * @param {(WebGLCoordinateSystem|WebGPUCoordinateSystem)} [coordinateSystem=WebGLCoordinateSystem] - The coordinate system.
	 * @param {boolean} [reversedDepth=false] - Whether to use a reversed depth.
	 * @return {Matrix4} A reference to this matrix.
	 */
  makeOrthographic(e, t, i, s, r, o, a = Nn, l = !1) {
    const c = this.elements, u = 2 / (t - e), h = 2 / (i - s), f = -(t + e) / (t - e), p = -(i + s) / (i - s);
    let v, x;
    if (l)
      v = 1 / (o - r), x = o / (o - r);
    else if (a === Nn)
      v = -2 / (o - r), x = -(o + r) / (o - r);
    else if (a === zo)
      v = -1 / (o - r), x = -r / (o - r);
    else
      throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: " + a);
    return c[0] = u, c[4] = 0, c[8] = 0, c[12] = f, c[1] = 0, c[5] = h, c[9] = 0, c[13] = p, c[2] = 0, c[6] = 0, c[10] = v, c[14] = x, c[3] = 0, c[7] = 0, c[11] = 0, c[15] = 1, this;
  }
  /**
   * Returns `true` if this matrix is equal with the given one.
   *
   * @param {Matrix4} matrix - The matrix to test for equality.
   * @return {boolean} Whether this matrix is equal with the given one.
   */
  equals(e) {
    const t = this.elements, i = e.elements;
    for (let s = 0; s < 16; s++)
      if (t[s] !== i[s]) return !1;
    return !0;
  }
  /**
   * Sets the elements of the matrix from the given array.
   *
   * @param {Array<number>} array - The matrix elements in column-major order.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Matrix4} A reference to this matrix.
   */
  fromArray(e, t = 0) {
    for (let i = 0; i < 16; i++)
      this.elements[i] = e[i + t];
    return this;
  }
  /**
   * Writes the elements of this matrix to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the matrix elements in column-major order.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The matrix elements in column-major order.
   */
  toArray(e = [], t = 0) {
    const i = this.elements;
    return e[t] = i[0], e[t + 1] = i[1], e[t + 2] = i[2], e[t + 3] = i[3], e[t + 4] = i[4], e[t + 5] = i[5], e[t + 6] = i[6], e[t + 7] = i[7], e[t + 8] = i[8], e[t + 9] = i[9], e[t + 10] = i[10], e[t + 11] = i[11], e[t + 12] = i[12], e[t + 13] = i[13], e[t + 14] = i[14], e[t + 15] = i[15], e;
  }
}
const cs = /* @__PURE__ */ new N(), gn = /* @__PURE__ */ new pt(), kg = /* @__PURE__ */ new N(0, 0, 0), Gg = /* @__PURE__ */ new N(1, 1, 1), ui = /* @__PURE__ */ new N(), jr = /* @__PURE__ */ new N(), en = /* @__PURE__ */ new N(), Ou = /* @__PURE__ */ new pt(), Bu = /* @__PURE__ */ new qi();
class zn {
  /**
   * Constructs a new euler instance.
   *
   * @param {number} [x=0] - The angle of the x axis in radians.
   * @param {number} [y=0] - The angle of the y axis in radians.
   * @param {number} [z=0] - The angle of the z axis in radians.
   * @param {string} [order=Euler.DEFAULT_ORDER] - A string representing the order that the rotations are applied.
   */
  constructor(e = 0, t = 0, i = 0, s = zn.DEFAULT_ORDER) {
    this.isEuler = !0, this._x = e, this._y = t, this._z = i, this._order = s;
  }
  /**
   * The angle of the x axis in radians.
   *
   * @type {number}
   * @default 0
   */
  get x() {
    return this._x;
  }
  set x(e) {
    this._x = e, this._onChangeCallback();
  }
  /**
   * The angle of the y axis in radians.
   *
   * @type {number}
   * @default 0
   */
  get y() {
    return this._y;
  }
  set y(e) {
    this._y = e, this._onChangeCallback();
  }
  /**
   * The angle of the z axis in radians.
   *
   * @type {number}
   * @default 0
   */
  get z() {
    return this._z;
  }
  set z(e) {
    this._z = e, this._onChangeCallback();
  }
  /**
   * A string representing the order that the rotations are applied.
   *
   * @type {string}
   * @default 'XYZ'
   */
  get order() {
    return this._order;
  }
  set order(e) {
    this._order = e, this._onChangeCallback();
  }
  /**
   * Sets the Euler components.
   *
   * @param {number} x - The angle of the x axis in radians.
   * @param {number} y - The angle of the y axis in radians.
   * @param {number} z - The angle of the z axis in radians.
   * @param {string} [order] - A string representing the order that the rotations are applied.
   * @return {Euler} A reference to this Euler instance.
   */
  set(e, t, i, s = this._order) {
    return this._x = e, this._y = t, this._z = i, this._order = s, this._onChangeCallback(), this;
  }
  /**
   * Returns a new Euler instance with copied values from this instance.
   *
   * @return {Euler} A clone of this instance.
   */
  clone() {
    return new this.constructor(this._x, this._y, this._z, this._order);
  }
  /**
   * Copies the values of the given Euler instance to this instance.
   *
   * @param {Euler} euler - The Euler instance to copy.
   * @return {Euler} A reference to this Euler instance.
   */
  copy(e) {
    return this._x = e._x, this._y = e._y, this._z = e._z, this._order = e._order, this._onChangeCallback(), this;
  }
  /**
   * Sets the angles of this Euler instance from a pure rotation matrix.
   *
   * @param {Matrix4} m - A 4x4 matrix of which the upper 3x3 of matrix is a pure rotation matrix (i.e. unscaled).
   * @param {string} [order] - A string representing the order that the rotations are applied.
   * @param {boolean} [update=true] - Whether the internal `onChange` callback should be executed or not.
   * @return {Euler} A reference to this Euler instance.
   */
  setFromRotationMatrix(e, t = this._order, i = !0) {
    const s = e.elements, r = s[0], o = s[4], a = s[8], l = s[1], c = s[5], u = s[9], h = s[2], f = s[6], p = s[10];
    switch (t) {
      case "XYZ":
        this._y = Math.asin(Ke(a, -1, 1)), Math.abs(a) < 0.9999999 ? (this._x = Math.atan2(-u, p), this._z = Math.atan2(-o, r)) : (this._x = Math.atan2(f, c), this._z = 0);
        break;
      case "YXZ":
        this._x = Math.asin(-Ke(u, -1, 1)), Math.abs(u) < 0.9999999 ? (this._y = Math.atan2(a, p), this._z = Math.atan2(l, c)) : (this._y = Math.atan2(-h, r), this._z = 0);
        break;
      case "ZXY":
        this._x = Math.asin(Ke(f, -1, 1)), Math.abs(f) < 0.9999999 ? (this._y = Math.atan2(-h, p), this._z = Math.atan2(-o, c)) : (this._y = 0, this._z = Math.atan2(l, r));
        break;
      case "ZYX":
        this._y = Math.asin(-Ke(h, -1, 1)), Math.abs(h) < 0.9999999 ? (this._x = Math.atan2(f, p), this._z = Math.atan2(l, r)) : (this._x = 0, this._z = Math.atan2(-o, c));
        break;
      case "YZX":
        this._z = Math.asin(Ke(l, -1, 1)), Math.abs(l) < 0.9999999 ? (this._x = Math.atan2(-u, c), this._y = Math.atan2(-h, r)) : (this._x = 0, this._y = Math.atan2(a, p));
        break;
      case "XZY":
        this._z = Math.asin(-Ke(o, -1, 1)), Math.abs(o) < 0.9999999 ? (this._x = Math.atan2(f, c), this._y = Math.atan2(a, r)) : (this._x = Math.atan2(-u, p), this._y = 0);
        break;
      default:
        console.warn("THREE.Euler: .setFromRotationMatrix() encountered an unknown order: " + t);
    }
    return this._order = t, i === !0 && this._onChangeCallback(), this;
  }
  /**
   * Sets the angles of this Euler instance from a normalized quaternion.
   *
   * @param {Quaternion} q - A normalized Quaternion.
   * @param {string} [order] - A string representing the order that the rotations are applied.
   * @param {boolean} [update=true] - Whether the internal `onChange` callback should be executed or not.
   * @return {Euler} A reference to this Euler instance.
   */
  setFromQuaternion(e, t, i) {
    return Ou.makeRotationFromQuaternion(e), this.setFromRotationMatrix(Ou, t, i);
  }
  /**
   * Sets the angles of this Euler instance from the given vector.
   *
   * @param {Vector3} v - The vector.
   * @param {string} [order] - A string representing the order that the rotations are applied.
   * @return {Euler} A reference to this Euler instance.
   */
  setFromVector3(e, t = this._order) {
    return this.set(e.x, e.y, e.z, t);
  }
  /**
   * Resets the euler angle with a new order by creating a quaternion from this
   * euler angle and then setting this euler angle with the quaternion and the
   * new order.
   *
   * Warning: This discards revolution information.
   *
   * @param {string} [newOrder] - A string representing the new order that the rotations are applied.
   * @return {Euler} A reference to this Euler instance.
   */
  reorder(e) {
    return Bu.setFromEuler(this), this.setFromQuaternion(Bu, e);
  }
  /**
   * Returns `true` if this Euler instance is equal with the given one.
   *
   * @param {Euler} euler - The Euler instance to test for equality.
   * @return {boolean} Whether this Euler instance is equal with the given one.
   */
  equals(e) {
    return e._x === this._x && e._y === this._y && e._z === this._z && e._order === this._order;
  }
  /**
   * Sets this Euler instance's components to values from the given array. The first three
   * entries of the array are assign to the x,y and z components. An optional fourth entry
   * defines the Euler order.
   *
   * @param {Array<number,number,number,?string>} array - An array holding the Euler component values.
   * @return {Euler} A reference to this Euler instance.
   */
  fromArray(e) {
    return this._x = e[0], this._y = e[1], this._z = e[2], e[3] !== void 0 && (this._order = e[3]), this._onChangeCallback(), this;
  }
  /**
   * Writes the components of this Euler instance to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number,number,number,string>} [array=[]] - The target array holding the Euler components.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number,number,number,string>} The Euler components.
   */
  toArray(e = [], t = 0) {
    return e[t] = this._x, e[t + 1] = this._y, e[t + 2] = this._z, e[t + 3] = this._order, e;
  }
  _onChange(e) {
    return this._onChangeCallback = e, this;
  }
  _onChangeCallback() {
  }
  *[Symbol.iterator]() {
    yield this._x, yield this._y, yield this._z, yield this._order;
  }
}
zn.DEFAULT_ORDER = "XYZ";
class _d {
  /**
   * Constructs a new layers instance, with membership
   * initially set to layer `0`.
   */
  constructor() {
    this.mask = 1;
  }
  /**
   * Sets membership to the given layer, and remove membership all other layers.
   *
   * @param {number} layer - The layer to set.
   */
  set(e) {
    this.mask = (1 << e | 0) >>> 0;
  }
  /**
   * Adds membership of the given layer.
   *
   * @param {number} layer - The layer to enable.
   */
  enable(e) {
    this.mask |= 1 << e | 0;
  }
  /**
   * Adds membership to all layers.
   */
  enableAll() {
    this.mask = -1;
  }
  /**
   * Toggles the membership of the given layer.
   *
   * @param {number} layer - The layer to toggle.
   */
  toggle(e) {
    this.mask ^= 1 << e | 0;
  }
  /**
   * Removes membership of the given layer.
   *
   * @param {number} layer - The layer to enable.
   */
  disable(e) {
    this.mask &= ~(1 << e | 0);
  }
  /**
   * Removes the membership from all layers.
   */
  disableAll() {
    this.mask = 0;
  }
  /**
   * Returns `true` if this and the given layers object have at least one
   * layer in common.
   *
   * @param {Layers} layers - The layers to test.
   * @return {boolean } Whether this and the given layers object have at least one layer in common or not.
   */
  test(e) {
    return (this.mask & e.mask) !== 0;
  }
  /**
   * Returns `true` if the given layer is enabled.
   *
   * @param {number} layer - The layer to test.
   * @return {boolean } Whether the given layer is enabled or not.
   */
  isEnabled(e) {
    return (this.mask & (1 << e | 0)) !== 0;
  }
}
let Wg = 0;
const zu = /* @__PURE__ */ new N(), us = /* @__PURE__ */ new qi(), Xn = /* @__PURE__ */ new pt(), Kr = /* @__PURE__ */ new N(), Js = /* @__PURE__ */ new N(), Xg = /* @__PURE__ */ new N(), Yg = /* @__PURE__ */ new qi(), Hu = /* @__PURE__ */ new N(1, 0, 0), Vu = /* @__PURE__ */ new N(0, 1, 0), ku = /* @__PURE__ */ new N(0, 0, 1), Gu = { type: "added" }, qg = { type: "removed" }, hs = { type: "childadded", child: null }, Pa = { type: "childremoved", child: null };
class Tt extends Ji {
  /**
   * Constructs a new 3D object.
   */
  constructor() {
    super(), this.isObject3D = !0, Object.defineProperty(this, "id", { value: Wg++ }), this.uuid = Ur(), this.name = "", this.type = "Object3D", this.parent = null, this.children = [], this.up = Tt.DEFAULT_UP.clone();
    const e = new N(), t = new zn(), i = new qi(), s = new N(1, 1, 1);
    function r() {
      i.setFromEuler(t, !1);
    }
    function o() {
      t.setFromQuaternion(i, void 0, !1);
    }
    t._onChange(r), i._onChange(o), Object.defineProperties(this, {
      /**
       * Represents the object's local position.
       *
       * @name Object3D#position
       * @type {Vector3}
       * @default (0,0,0)
       */
      position: {
        configurable: !0,
        enumerable: !0,
        value: e
      },
      /**
       * Represents the object's local rotation as Euler angles, in radians.
       *
       * @name Object3D#rotation
       * @type {Euler}
       * @default (0,0,0)
       */
      rotation: {
        configurable: !0,
        enumerable: !0,
        value: t
      },
      /**
       * Represents the object's local rotation as Quaternions.
       *
       * @name Object3D#quaternion
       * @type {Quaternion}
       */
      quaternion: {
        configurable: !0,
        enumerable: !0,
        value: i
      },
      /**
       * Represents the object's local scale.
       *
       * @name Object3D#scale
       * @type {Vector3}
       * @default (1,1,1)
       */
      scale: {
        configurable: !0,
        enumerable: !0,
        value: s
      },
      /**
       * Represents the object's model-view matrix.
       *
       * @name Object3D#modelViewMatrix
       * @type {Matrix4}
       */
      modelViewMatrix: {
        value: new pt()
      },
      /**
       * Represents the object's normal matrix.
       *
       * @name Object3D#normalMatrix
       * @type {Matrix3}
       */
      normalMatrix: {
        value: new qe()
      }
    }), this.matrix = new pt(), this.matrixWorld = new pt(), this.matrixAutoUpdate = Tt.DEFAULT_MATRIX_AUTO_UPDATE, this.matrixWorldAutoUpdate = Tt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE, this.matrixWorldNeedsUpdate = !1, this.layers = new _d(), this.visible = !0, this.castShadow = !1, this.receiveShadow = !1, this.frustumCulled = !0, this.renderOrder = 0, this.animations = [], this.customDepthMaterial = void 0, this.customDistanceMaterial = void 0, this.userData = {};
  }
  /**
   * A callback that is executed immediately before a 3D object is rendered to a shadow map.
   *
   * @param {Renderer|WebGLRenderer} renderer - The renderer.
   * @param {Object3D} object - The 3D object.
   * @param {Camera} camera - The camera that is used to render the scene.
   * @param {Camera} shadowCamera - The shadow camera.
   * @param {BufferGeometry} geometry - The 3D object's geometry.
   * @param {Material} depthMaterial - The depth material.
   * @param {Object} group - The geometry group data.
   */
  onBeforeShadow() {
  }
  /**
   * A callback that is executed immediately after a 3D object is rendered to a shadow map.
   *
   * @param {Renderer|WebGLRenderer} renderer - The renderer.
   * @param {Object3D} object - The 3D object.
   * @param {Camera} camera - The camera that is used to render the scene.
   * @param {Camera} shadowCamera - The shadow camera.
   * @param {BufferGeometry} geometry - The 3D object's geometry.
   * @param {Material} depthMaterial - The depth material.
   * @param {Object} group - The geometry group data.
   */
  onAfterShadow() {
  }
  /**
   * A callback that is executed immediately before a 3D object is rendered.
   *
   * @param {Renderer|WebGLRenderer} renderer - The renderer.
   * @param {Object3D} object - The 3D object.
   * @param {Camera} camera - The camera that is used to render the scene.
   * @param {BufferGeometry} geometry - The 3D object's geometry.
   * @param {Material} material - The 3D object's material.
   * @param {Object} group - The geometry group data.
   */
  onBeforeRender() {
  }
  /**
   * A callback that is executed immediately after a 3D object is rendered.
   *
   * @param {Renderer|WebGLRenderer} renderer - The renderer.
   * @param {Object3D} object - The 3D object.
   * @param {Camera} camera - The camera that is used to render the scene.
   * @param {BufferGeometry} geometry - The 3D object's geometry.
   * @param {Material} material - The 3D object's material.
   * @param {Object} group - The geometry group data.
   */
  onAfterRender() {
  }
  /**
   * Applies the given transformation matrix to the object and updates the object's position,
   * rotation and scale.
   *
   * @param {Matrix4} matrix - The transformation matrix.
   */
  applyMatrix4(e) {
    this.matrixAutoUpdate && this.updateMatrix(), this.matrix.premultiply(e), this.matrix.decompose(this.position, this.quaternion, this.scale);
  }
  /**
   * Applies a rotation represented by given the quaternion to the 3D object.
   *
   * @param {Quaternion} q - The quaternion.
   * @return {Object3D} A reference to this instance.
   */
  applyQuaternion(e) {
    return this.quaternion.premultiply(e), this;
  }
  /**
   * Sets the given rotation represented as an axis/angle couple to the 3D object.
   *
   * @param {Vector3} axis - The (normalized) axis vector.
   * @param {number} angle - The angle in radians.
   */
  setRotationFromAxisAngle(e, t) {
    this.quaternion.setFromAxisAngle(e, t);
  }
  /**
   * Sets the given rotation represented as Euler angles to the 3D object.
   *
   * @param {Euler} euler - The Euler angles.
   */
  setRotationFromEuler(e) {
    this.quaternion.setFromEuler(e, !0);
  }
  /**
   * Sets the given rotation represented as rotation matrix to the 3D object.
   *
   * @param {Matrix4} m - Although a 4x4 matrix is expected, the upper 3x3 portion must be
   * a pure rotation matrix (i.e, unscaled).
   */
  setRotationFromMatrix(e) {
    this.quaternion.setFromRotationMatrix(e);
  }
  /**
   * Sets the given rotation represented as a Quaternion to the 3D object.
   *
   * @param {Quaternion} q - The Quaternion
   */
  setRotationFromQuaternion(e) {
    this.quaternion.copy(e);
  }
  /**
   * Rotates the 3D object along an axis in local space.
   *
   * @param {Vector3} axis - The (normalized) axis vector.
   * @param {number} angle - The angle in radians.
   * @return {Object3D} A reference to this instance.
   */
  rotateOnAxis(e, t) {
    return us.setFromAxisAngle(e, t), this.quaternion.multiply(us), this;
  }
  /**
   * Rotates the 3D object along an axis in world space.
   *
   * @param {Vector3} axis - The (normalized) axis vector.
   * @param {number} angle - The angle in radians.
   * @return {Object3D} A reference to this instance.
   */
  rotateOnWorldAxis(e, t) {
    return us.setFromAxisAngle(e, t), this.quaternion.premultiply(us), this;
  }
  /**
   * Rotates the 3D object around its X axis in local space.
   *
   * @param {number} angle - The angle in radians.
   * @return {Object3D} A reference to this instance.
   */
  rotateX(e) {
    return this.rotateOnAxis(Hu, e);
  }
  /**
   * Rotates the 3D object around its Y axis in local space.
   *
   * @param {number} angle - The angle in radians.
   * @return {Object3D} A reference to this instance.
   */
  rotateY(e) {
    return this.rotateOnAxis(Vu, e);
  }
  /**
   * Rotates the 3D object around its Z axis in local space.
   *
   * @param {number} angle - The angle in radians.
   * @return {Object3D} A reference to this instance.
   */
  rotateZ(e) {
    return this.rotateOnAxis(ku, e);
  }
  /**
   * Translate the 3D object by a distance along the given axis in local space.
   *
   * @param {Vector3} axis - The (normalized) axis vector.
   * @param {number} distance - The distance in world units.
   * @return {Object3D} A reference to this instance.
   */
  translateOnAxis(e, t) {
    return zu.copy(e).applyQuaternion(this.quaternion), this.position.add(zu.multiplyScalar(t)), this;
  }
  /**
   * Translate the 3D object by a distance along its X-axis in local space.
   *
   * @param {number} distance - The distance in world units.
   * @return {Object3D} A reference to this instance.
   */
  translateX(e) {
    return this.translateOnAxis(Hu, e);
  }
  /**
   * Translate the 3D object by a distance along its Y-axis in local space.
   *
   * @param {number} distance - The distance in world units.
   * @return {Object3D} A reference to this instance.
   */
  translateY(e) {
    return this.translateOnAxis(Vu, e);
  }
  /**
   * Translate the 3D object by a distance along its Z-axis in local space.
   *
   * @param {number} distance - The distance in world units.
   * @return {Object3D} A reference to this instance.
   */
  translateZ(e) {
    return this.translateOnAxis(ku, e);
  }
  /**
   * Converts the given vector from this 3D object's local space to world space.
   *
   * @param {Vector3} vector - The vector to convert.
   * @return {Vector3} The converted vector.
   */
  localToWorld(e) {
    return this.updateWorldMatrix(!0, !1), e.applyMatrix4(this.matrixWorld);
  }
  /**
   * Converts the given vector from this 3D object's word space to local space.
   *
   * @param {Vector3} vector - The vector to convert.
   * @return {Vector3} The converted vector.
   */
  worldToLocal(e) {
    return this.updateWorldMatrix(!0, !1), e.applyMatrix4(Xn.copy(this.matrixWorld).invert());
  }
  /**
   * Rotates the object to face a point in world space.
   *
   * This method does not support objects having non-uniformly-scaled parent(s).
   *
   * @param {number|Vector3} x - The x coordinate in world space. Alternatively, a vector representing a position in world space
   * @param {number} [y] - The y coordinate in world space.
   * @param {number} [z] - The z coordinate in world space.
   */
  lookAt(e, t, i) {
    e.isVector3 ? Kr.copy(e) : Kr.set(e, t, i);
    const s = this.parent;
    this.updateWorldMatrix(!0, !1), Js.setFromMatrixPosition(this.matrixWorld), this.isCamera || this.isLight ? Xn.lookAt(Js, Kr, this.up) : Xn.lookAt(Kr, Js, this.up), this.quaternion.setFromRotationMatrix(Xn), s && (Xn.extractRotation(s.matrixWorld), us.setFromRotationMatrix(Xn), this.quaternion.premultiply(us.invert()));
  }
  /**
   * Adds the given 3D object as a child to this 3D object. An arbitrary number of
   * objects may be added. Any current parent on an object passed in here will be
   * removed, since an object can have at most one parent.
   *
   * @fires Object3D#added
   * @fires Object3D#childadded
   * @param {Object3D} object - The 3D object to add.
   * @return {Object3D} A reference to this instance.
   */
  add(e) {
    if (arguments.length > 1) {
      for (let t = 0; t < arguments.length; t++)
        this.add(arguments[t]);
      return this;
    }
    return e === this ? (console.error("THREE.Object3D.add: object can't be added as a child of itself.", e), this) : (e && e.isObject3D ? (e.removeFromParent(), e.parent = this, this.children.push(e), e.dispatchEvent(Gu), hs.child = e, this.dispatchEvent(hs), hs.child = null) : console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.", e), this);
  }
  /**
   * Removes the given 3D object as child from this 3D object.
   * An arbitrary number of objects may be removed.
   *
   * @fires Object3D#removed
   * @fires Object3D#childremoved
   * @param {Object3D} object - The 3D object to remove.
   * @return {Object3D} A reference to this instance.
   */
  remove(e) {
    if (arguments.length > 1) {
      for (let i = 0; i < arguments.length; i++)
        this.remove(arguments[i]);
      return this;
    }
    const t = this.children.indexOf(e);
    return t !== -1 && (e.parent = null, this.children.splice(t, 1), e.dispatchEvent(qg), Pa.child = e, this.dispatchEvent(Pa), Pa.child = null), this;
  }
  /**
   * Removes this 3D object from its current parent.
   *
   * @fires Object3D#removed
   * @fires Object3D#childremoved
   * @return {Object3D} A reference to this instance.
   */
  removeFromParent() {
    const e = this.parent;
    return e !== null && e.remove(this), this;
  }
  /**
   * Removes all child objects.
   *
   * @fires Object3D#removed
   * @fires Object3D#childremoved
   * @return {Object3D} A reference to this instance.
   */
  clear() {
    return this.remove(...this.children);
  }
  /**
   * Adds the given 3D object as a child of this 3D object, while maintaining the object's world
   * transform. This method does not support scene graphs having non-uniformly-scaled nodes(s).
   *
   * @fires Object3D#added
   * @fires Object3D#childadded
   * @param {Object3D} object - The 3D object to attach.
   * @return {Object3D} A reference to this instance.
   */
  attach(e) {
    return this.updateWorldMatrix(!0, !1), Xn.copy(this.matrixWorld).invert(), e.parent !== null && (e.parent.updateWorldMatrix(!0, !1), Xn.multiply(e.parent.matrixWorld)), e.applyMatrix4(Xn), e.removeFromParent(), e.parent = this, this.children.push(e), e.updateWorldMatrix(!1, !0), e.dispatchEvent(Gu), hs.child = e, this.dispatchEvent(hs), hs.child = null, this;
  }
  /**
   * Searches through the 3D object and its children, starting with the 3D object
   * itself, and returns the first with a matching ID.
   *
   * @param {number} id - The id.
   * @return {Object3D|undefined} The found 3D object. Returns `undefined` if no 3D object has been found.
   */
  getObjectById(e) {
    return this.getObjectByProperty("id", e);
  }
  /**
   * Searches through the 3D object and its children, starting with the 3D object
   * itself, and returns the first with a matching name.
   *
   * @param {string} name - The name.
   * @return {Object3D|undefined} The found 3D object. Returns `undefined` if no 3D object has been found.
   */
  getObjectByName(e) {
    return this.getObjectByProperty("name", e);
  }
  /**
   * Searches through the 3D object and its children, starting with the 3D object
   * itself, and returns the first with a matching property value.
   *
   * @param {string} name - The name of the property.
   * @param {any} value - The value.
   * @return {Object3D|undefined} The found 3D object. Returns `undefined` if no 3D object has been found.
   */
  getObjectByProperty(e, t) {
    if (this[e] === t) return this;
    for (let i = 0, s = this.children.length; i < s; i++) {
      const o = this.children[i].getObjectByProperty(e, t);
      if (o !== void 0)
        return o;
    }
  }
  /**
   * Searches through the 3D object and its children, starting with the 3D object
   * itself, and returns all 3D objects with a matching property value.
   *
   * @param {string} name - The name of the property.
   * @param {any} value - The value.
   * @param {Array<Object3D>} result - The method stores the result in this array.
   * @return {Array<Object3D>} The found 3D objects.
   */
  getObjectsByProperty(e, t, i = []) {
    this[e] === t && i.push(this);
    const s = this.children;
    for (let r = 0, o = s.length; r < o; r++)
      s[r].getObjectsByProperty(e, t, i);
    return i;
  }
  /**
   * Returns a vector representing the position of the 3D object in world space.
   *
   * @param {Vector3} target - The target vector the result is stored to.
   * @return {Vector3} The 3D object's position in world space.
   */
  getWorldPosition(e) {
    return this.updateWorldMatrix(!0, !1), e.setFromMatrixPosition(this.matrixWorld);
  }
  /**
   * Returns a Quaternion representing the position of the 3D object in world space.
   *
   * @param {Quaternion} target - The target Quaternion the result is stored to.
   * @return {Quaternion} The 3D object's rotation in world space.
   */
  getWorldQuaternion(e) {
    return this.updateWorldMatrix(!0, !1), this.matrixWorld.decompose(Js, e, Xg), e;
  }
  /**
   * Returns a vector representing the scale of the 3D object in world space.
   *
   * @param {Vector3} target - The target vector the result is stored to.
   * @return {Vector3} The 3D object's scale in world space.
   */
  getWorldScale(e) {
    return this.updateWorldMatrix(!0, !1), this.matrixWorld.decompose(Js, Yg, e), e;
  }
  /**
   * Returns a vector representing the ("look") direction of the 3D object in world space.
   *
   * @param {Vector3} target - The target vector the result is stored to.
   * @return {Vector3} The 3D object's direction in world space.
   */
  getWorldDirection(e) {
    this.updateWorldMatrix(!0, !1);
    const t = this.matrixWorld.elements;
    return e.set(t[8], t[9], t[10]).normalize();
  }
  /**
   * Abstract method to get intersections between a casted ray and this
   * 3D object. Renderable 3D objects such as {@link Mesh}, {@link Line} or {@link Points}
   * implement this method in order to use raycasting.
   *
   * @abstract
   * @param {Raycaster} raycaster - The raycaster.
   * @param {Array<Object>} intersects - An array holding the result of the method.
   */
  raycast() {
  }
  /**
   * Executes the callback on this 3D object and all descendants.
   *
   * Note: Modifying the scene graph inside the callback is discouraged.
   *
   * @param {Function} callback - A callback function that allows to process the current 3D object.
   */
  traverse(e) {
    e(this);
    const t = this.children;
    for (let i = 0, s = t.length; i < s; i++)
      t[i].traverse(e);
  }
  /**
   * Like {@link Object3D#traverse}, but the callback will only be executed for visible 3D objects.
   * Descendants of invisible 3D objects are not traversed.
   *
   * Note: Modifying the scene graph inside the callback is discouraged.
   *
   * @param {Function} callback - A callback function that allows to process the current 3D object.
   */
  traverseVisible(e) {
    if (this.visible === !1) return;
    e(this);
    const t = this.children;
    for (let i = 0, s = t.length; i < s; i++)
      t[i].traverseVisible(e);
  }
  /**
   * Like {@link Object3D#traverse}, but the callback will only be executed for all ancestors.
   *
   * Note: Modifying the scene graph inside the callback is discouraged.
   *
   * @param {Function} callback - A callback function that allows to process the current 3D object.
   */
  traverseAncestors(e) {
    const t = this.parent;
    t !== null && (e(t), t.traverseAncestors(e));
  }
  /**
   * Updates the transformation matrix in local space by computing it from the current
   * position, rotation and scale values.
   */
  updateMatrix() {
    this.matrix.compose(this.position, this.quaternion, this.scale), this.matrixWorldNeedsUpdate = !0;
  }
  /**
   * Updates the transformation matrix in world space of this 3D objects and its descendants.
   *
   * To ensure correct results, this method also recomputes the 3D object's transformation matrix in
   * local space. The computation of the local and world matrix can be controlled with the
   * {@link Object3D#matrixAutoUpdate} and {@link Object3D#matrixWorldAutoUpdate} flags which are both
   * `true` by default.  Set these flags to `false` if you need more control over the update matrix process.
   *
   * @param {boolean} [force=false] - When set to `true`, a recomputation of world matrices is forced even
   * when {@link Object3D#matrixWorldAutoUpdate} is set to `false`.
   */
  updateMatrixWorld(e) {
    this.matrixAutoUpdate && this.updateMatrix(), (this.matrixWorldNeedsUpdate || e) && (this.matrixWorldAutoUpdate === !0 && (this.parent === null ? this.matrixWorld.copy(this.matrix) : this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix)), this.matrixWorldNeedsUpdate = !1, e = !0);
    const t = this.children;
    for (let i = 0, s = t.length; i < s; i++)
      t[i].updateMatrixWorld(e);
  }
  /**
   * An alternative version of {@link Object3D#updateMatrixWorld} with more control over the
   * update of ancestor and descendant nodes.
   *
   * @param {boolean} [updateParents=false] Whether ancestor nodes should be updated or not.
   * @param {boolean} [updateChildren=false] Whether descendant nodes should be updated or not.
   */
  updateWorldMatrix(e, t) {
    const i = this.parent;
    if (e === !0 && i !== null && i.updateWorldMatrix(!0, !1), this.matrixAutoUpdate && this.updateMatrix(), this.matrixWorldAutoUpdate === !0 && (this.parent === null ? this.matrixWorld.copy(this.matrix) : this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix)), t === !0) {
      const s = this.children;
      for (let r = 0, o = s.length; r < o; r++)
        s[r].updateWorldMatrix(!1, !0);
    }
  }
  /**
   * Serializes the 3D object into JSON.
   *
   * @param {?(Object|string)} meta - An optional value holding meta information about the serialization.
   * @return {Object} A JSON object representing the serialized 3D object.
   * @see {@link ObjectLoader#parse}
   */
  toJSON(e) {
    const t = e === void 0 || typeof e == "string", i = {};
    t && (e = {
      geometries: {},
      materials: {},
      textures: {},
      images: {},
      shapes: {},
      skeletons: {},
      animations: {},
      nodes: {}
    }, i.metadata = {
      version: 4.7,
      type: "Object",
      generator: "Object3D.toJSON"
    });
    const s = {};
    s.uuid = this.uuid, s.type = this.type, this.name !== "" && (s.name = this.name), this.castShadow === !0 && (s.castShadow = !0), this.receiveShadow === !0 && (s.receiveShadow = !0), this.visible === !1 && (s.visible = !1), this.frustumCulled === !1 && (s.frustumCulled = !1), this.renderOrder !== 0 && (s.renderOrder = this.renderOrder), Object.keys(this.userData).length > 0 && (s.userData = this.userData), s.layers = this.layers.mask, s.matrix = this.matrix.toArray(), s.up = this.up.toArray(), this.matrixAutoUpdate === !1 && (s.matrixAutoUpdate = !1), this.isInstancedMesh && (s.type = "InstancedMesh", s.count = this.count, s.instanceMatrix = this.instanceMatrix.toJSON(), this.instanceColor !== null && (s.instanceColor = this.instanceColor.toJSON())), this.isBatchedMesh && (s.type = "BatchedMesh", s.perObjectFrustumCulled = this.perObjectFrustumCulled, s.sortObjects = this.sortObjects, s.drawRanges = this._drawRanges, s.reservedRanges = this._reservedRanges, s.geometryInfo = this._geometryInfo.map((a) => ({
      ...a,
      boundingBox: a.boundingBox ? a.boundingBox.toJSON() : void 0,
      boundingSphere: a.boundingSphere ? a.boundingSphere.toJSON() : void 0
    })), s.instanceInfo = this._instanceInfo.map((a) => ({ ...a })), s.availableInstanceIds = this._availableInstanceIds.slice(), s.availableGeometryIds = this._availableGeometryIds.slice(), s.nextIndexStart = this._nextIndexStart, s.nextVertexStart = this._nextVertexStart, s.geometryCount = this._geometryCount, s.maxInstanceCount = this._maxInstanceCount, s.maxVertexCount = this._maxVertexCount, s.maxIndexCount = this._maxIndexCount, s.geometryInitialized = this._geometryInitialized, s.matricesTexture = this._matricesTexture.toJSON(e), s.indirectTexture = this._indirectTexture.toJSON(e), this._colorsTexture !== null && (s.colorsTexture = this._colorsTexture.toJSON(e)), this.boundingSphere !== null && (s.boundingSphere = this.boundingSphere.toJSON()), this.boundingBox !== null && (s.boundingBox = this.boundingBox.toJSON()));
    function r(a, l) {
      return a[l.uuid] === void 0 && (a[l.uuid] = l.toJSON(e)), l.uuid;
    }
    if (this.isScene)
      this.background && (this.background.isColor ? s.background = this.background.toJSON() : this.background.isTexture && (s.background = this.background.toJSON(e).uuid)), this.environment && this.environment.isTexture && this.environment.isRenderTargetTexture !== !0 && (s.environment = this.environment.toJSON(e).uuid);
    else if (this.isMesh || this.isLine || this.isPoints) {
      s.geometry = r(e.geometries, this.geometry);
      const a = this.geometry.parameters;
      if (a !== void 0 && a.shapes !== void 0) {
        const l = a.shapes;
        if (Array.isArray(l))
          for (let c = 0, u = l.length; c < u; c++) {
            const h = l[c];
            r(e.shapes, h);
          }
        else
          r(e.shapes, l);
      }
    }
    if (this.isSkinnedMesh && (s.bindMode = this.bindMode, s.bindMatrix = this.bindMatrix.toArray(), this.skeleton !== void 0 && (r(e.skeletons, this.skeleton), s.skeleton = this.skeleton.uuid)), this.material !== void 0)
      if (Array.isArray(this.material)) {
        const a = [];
        for (let l = 0, c = this.material.length; l < c; l++)
          a.push(r(e.materials, this.material[l]));
        s.material = a;
      } else
        s.material = r(e.materials, this.material);
    if (this.children.length > 0) {
      s.children = [];
      for (let a = 0; a < this.children.length; a++)
        s.children.push(this.children[a].toJSON(e).object);
    }
    if (this.animations.length > 0) {
      s.animations = [];
      for (let a = 0; a < this.animations.length; a++) {
        const l = this.animations[a];
        s.animations.push(r(e.animations, l));
      }
    }
    if (t) {
      const a = o(e.geometries), l = o(e.materials), c = o(e.textures), u = o(e.images), h = o(e.shapes), f = o(e.skeletons), p = o(e.animations), v = o(e.nodes);
      a.length > 0 && (i.geometries = a), l.length > 0 && (i.materials = l), c.length > 0 && (i.textures = c), u.length > 0 && (i.images = u), h.length > 0 && (i.shapes = h), f.length > 0 && (i.skeletons = f), p.length > 0 && (i.animations = p), v.length > 0 && (i.nodes = v);
    }
    return i.object = s, i;
    function o(a) {
      const l = [];
      for (const c in a) {
        const u = a[c];
        delete u.metadata, l.push(u);
      }
      return l;
    }
  }
  /**
   * Returns a new 3D object with copied values from this instance.
   *
   * @param {boolean} [recursive=true] - When set to `true`, descendants of the 3D object are also cloned.
   * @return {Object3D} A clone of this instance.
   */
  clone(e) {
    return new this.constructor().copy(this, e);
  }
  /**
   * Copies the values of the given 3D object to this instance.
   *
   * @param {Object3D} source - The 3D object to copy.
   * @param {boolean} [recursive=true] - When set to `true`, descendants of the 3D object are cloned.
   * @return {Object3D} A reference to this instance.
   */
  copy(e, t = !0) {
    if (this.name = e.name, this.up.copy(e.up), this.position.copy(e.position), this.rotation.order = e.rotation.order, this.quaternion.copy(e.quaternion), this.scale.copy(e.scale), this.matrix.copy(e.matrix), this.matrixWorld.copy(e.matrixWorld), this.matrixAutoUpdate = e.matrixAutoUpdate, this.matrixWorldAutoUpdate = e.matrixWorldAutoUpdate, this.matrixWorldNeedsUpdate = e.matrixWorldNeedsUpdate, this.layers.mask = e.layers.mask, this.visible = e.visible, this.castShadow = e.castShadow, this.receiveShadow = e.receiveShadow, this.frustumCulled = e.frustumCulled, this.renderOrder = e.renderOrder, this.animations = e.animations.slice(), this.userData = JSON.parse(JSON.stringify(e.userData)), t === !0)
      for (let i = 0; i < e.children.length; i++) {
        const s = e.children[i];
        this.add(s.clone());
      }
    return this;
  }
}
Tt.DEFAULT_UP = /* @__PURE__ */ new N(0, 1, 0);
Tt.DEFAULT_MATRIX_AUTO_UPDATE = !0;
Tt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE = !0;
const vn = /* @__PURE__ */ new N(), Yn = /* @__PURE__ */ new N(), Da = /* @__PURE__ */ new N(), qn = /* @__PURE__ */ new N(), fs = /* @__PURE__ */ new N(), ds = /* @__PURE__ */ new N(), Wu = /* @__PURE__ */ new N(), La = /* @__PURE__ */ new N(), Ia = /* @__PURE__ */ new N(), Ua = /* @__PURE__ */ new N(), Na = /* @__PURE__ */ new lt(), Fa = /* @__PURE__ */ new lt(), Oa = /* @__PURE__ */ new lt();
class fn {
  /**
   * Constructs a new triangle.
   *
   * @param {Vector3} [a=(0,0,0)] - The first corner of the triangle.
   * @param {Vector3} [b=(0,0,0)] - The second corner of the triangle.
   * @param {Vector3} [c=(0,0,0)] - The third corner of the triangle.
   */
  constructor(e = new N(), t = new N(), i = new N()) {
    this.a = e, this.b = t, this.c = i;
  }
  /**
   * Computes the normal vector of a triangle.
   *
   * @param {Vector3} a - The first corner of the triangle.
   * @param {Vector3} b - The second corner of the triangle.
   * @param {Vector3} c - The third corner of the triangle.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The triangle's normal.
   */
  static getNormal(e, t, i, s) {
    s.subVectors(i, t), vn.subVectors(e, t), s.cross(vn);
    const r = s.lengthSq();
    return r > 0 ? s.multiplyScalar(1 / Math.sqrt(r)) : s.set(0, 0, 0);
  }
  /**
   * Computes a barycentric coordinates from the given vector.
   * Returns `null` if the triangle is degenerate.
   *
   * @param {Vector3} point - A point in 3D space.
   * @param {Vector3} a - The first corner of the triangle.
   * @param {Vector3} b - The second corner of the triangle.
   * @param {Vector3} c - The third corner of the triangle.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The barycentric coordinates for the given point
   */
  static getBarycoord(e, t, i, s, r) {
    vn.subVectors(s, t), Yn.subVectors(i, t), Da.subVectors(e, t);
    const o = vn.dot(vn), a = vn.dot(Yn), l = vn.dot(Da), c = Yn.dot(Yn), u = Yn.dot(Da), h = o * c - a * a;
    if (h === 0)
      return r.set(0, 0, 0), null;
    const f = 1 / h, p = (c * l - a * u) * f, v = (o * u - a * l) * f;
    return r.set(1 - p - v, v, p);
  }
  /**
   * Returns `true` if the given point, when projected onto the plane of the
   * triangle, lies within the triangle.
   *
   * @param {Vector3} point - The point in 3D space to test.
   * @param {Vector3} a - The first corner of the triangle.
   * @param {Vector3} b - The second corner of the triangle.
   * @param {Vector3} c - The third corner of the triangle.
   * @return {boolean} Whether the given point, when projected onto the plane of the
   * triangle, lies within the triangle or not.
   */
  static containsPoint(e, t, i, s) {
    return this.getBarycoord(e, t, i, s, qn) === null ? !1 : qn.x >= 0 && qn.y >= 0 && qn.x + qn.y <= 1;
  }
  /**
   * Computes the value barycentrically interpolated for the given point on the
   * triangle. Returns `null` if the triangle is degenerate.
   *
   * @param {Vector3} point - Position of interpolated point.
   * @param {Vector3} p1 - The first corner of the triangle.
   * @param {Vector3} p2 - The second corner of the triangle.
   * @param {Vector3} p3 - The third corner of the triangle.
   * @param {Vector3} v1 - Value to interpolate of first vertex.
   * @param {Vector3} v2 - Value to interpolate of second vertex.
   * @param {Vector3} v3 - Value to interpolate of third vertex.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The interpolated value.
   */
  static getInterpolation(e, t, i, s, r, o, a, l) {
    return this.getBarycoord(e, t, i, s, qn) === null ? (l.x = 0, l.y = 0, "z" in l && (l.z = 0), "w" in l && (l.w = 0), null) : (l.setScalar(0), l.addScaledVector(r, qn.x), l.addScaledVector(o, qn.y), l.addScaledVector(a, qn.z), l);
  }
  /**
   * Computes the value barycentrically interpolated for the given attribute and indices.
   *
   * @param {BufferAttribute} attr - The attribute to interpolate.
   * @param {number} i1 - Index of first vertex.
   * @param {number} i2 - Index of second vertex.
   * @param {number} i3 - Index of third vertex.
   * @param {Vector3} barycoord - The barycoordinate value to use to interpolate.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The interpolated attribute value.
   */
  static getInterpolatedAttribute(e, t, i, s, r, o) {
    return Na.setScalar(0), Fa.setScalar(0), Oa.setScalar(0), Na.fromBufferAttribute(e, t), Fa.fromBufferAttribute(e, i), Oa.fromBufferAttribute(e, s), o.setScalar(0), o.addScaledVector(Na, r.x), o.addScaledVector(Fa, r.y), o.addScaledVector(Oa, r.z), o;
  }
  /**
   * Returns `true` if the triangle is oriented towards the given direction.
   *
   * @param {Vector3} a - The first corner of the triangle.
   * @param {Vector3} b - The second corner of the triangle.
   * @param {Vector3} c - The third corner of the triangle.
   * @param {Vector3} direction - The (normalized) direction vector.
   * @return {boolean} Whether the triangle is oriented towards the given direction or not.
   */
  static isFrontFacing(e, t, i, s) {
    return vn.subVectors(i, t), Yn.subVectors(e, t), vn.cross(Yn).dot(s) < 0;
  }
  /**
   * Sets the triangle's vertices by copying the given values.
   *
   * @param {Vector3} a - The first corner of the triangle.
   * @param {Vector3} b - The second corner of the triangle.
   * @param {Vector3} c - The third corner of the triangle.
   * @return {Triangle} A reference to this triangle.
   */
  set(e, t, i) {
    return this.a.copy(e), this.b.copy(t), this.c.copy(i), this;
  }
  /**
   * Sets the triangle's vertices by copying the given array values.
   *
   * @param {Array<Vector3>} points - An array with 3D points.
   * @param {number} i0 - The array index representing the first corner of the triangle.
   * @param {number} i1 - The array index representing the second corner of the triangle.
   * @param {number} i2 - The array index representing the third corner of the triangle.
   * @return {Triangle} A reference to this triangle.
   */
  setFromPointsAndIndices(e, t, i, s) {
    return this.a.copy(e[t]), this.b.copy(e[i]), this.c.copy(e[s]), this;
  }
  /**
   * Sets the triangle's vertices by copying the given attribute values.
   *
   * @param {BufferAttribute} attribute - A buffer attribute with 3D points data.
   * @param {number} i0 - The attribute index representing the first corner of the triangle.
   * @param {number} i1 - The attribute index representing the second corner of the triangle.
   * @param {number} i2 - The attribute index representing the third corner of the triangle.
   * @return {Triangle} A reference to this triangle.
   */
  setFromAttributeAndIndices(e, t, i, s) {
    return this.a.fromBufferAttribute(e, t), this.b.fromBufferAttribute(e, i), this.c.fromBufferAttribute(e, s), this;
  }
  /**
   * Returns a new triangle with copied values from this instance.
   *
   * @return {Triangle} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Copies the values of the given triangle to this instance.
   *
   * @param {Triangle} triangle - The triangle to copy.
   * @return {Triangle} A reference to this triangle.
   */
  copy(e) {
    return this.a.copy(e.a), this.b.copy(e.b), this.c.copy(e.c), this;
  }
  /**
   * Computes the area of the triangle.
   *
   * @return {number} The triangle's area.
   */
  getArea() {
    return vn.subVectors(this.c, this.b), Yn.subVectors(this.a, this.b), vn.cross(Yn).length() * 0.5;
  }
  /**
   * Computes the midpoint of the triangle.
   *
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The triangle's midpoint.
   */
  getMidpoint(e) {
    return e.addVectors(this.a, this.b).add(this.c).multiplyScalar(1 / 3);
  }
  /**
   * Computes the normal of the triangle.
   *
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The triangle's normal.
   */
  getNormal(e) {
    return fn.getNormal(this.a, this.b, this.c, e);
  }
  /**
   * Computes a plane the triangle lies within.
   *
   * @param {Plane} target - The target vector that is used to store the method's result.
   * @return {Plane} The plane the triangle lies within.
   */
  getPlane(e) {
    return e.setFromCoplanarPoints(this.a, this.b, this.c);
  }
  /**
   * Computes a barycentric coordinates from the given vector.
   * Returns `null` if the triangle is degenerate.
   *
   * @param {Vector3} point - A point in 3D space.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The barycentric coordinates for the given point
   */
  getBarycoord(e, t) {
    return fn.getBarycoord(e, this.a, this.b, this.c, t);
  }
  /**
   * Computes the value barycentrically interpolated for the given point on the
   * triangle. Returns `null` if the triangle is degenerate.
   *
   * @param {Vector3} point - Position of interpolated point.
   * @param {Vector3} v1 - Value to interpolate of first vertex.
   * @param {Vector3} v2 - Value to interpolate of second vertex.
   * @param {Vector3} v3 - Value to interpolate of third vertex.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The interpolated value.
   */
  getInterpolation(e, t, i, s, r) {
    return fn.getInterpolation(e, this.a, this.b, this.c, t, i, s, r);
  }
  /**
   * Returns `true` if the given point, when projected onto the plane of the
   * triangle, lies within the triangle.
   *
   * @param {Vector3} point - The point in 3D space to test.
   * @return {boolean} Whether the given point, when projected onto the plane of the
   * triangle, lies within the triangle or not.
   */
  containsPoint(e) {
    return fn.containsPoint(e, this.a, this.b, this.c);
  }
  /**
   * Returns `true` if the triangle is oriented towards the given direction.
   *
   * @param {Vector3} direction - The (normalized) direction vector.
   * @return {boolean} Whether the triangle is oriented towards the given direction or not.
   */
  isFrontFacing(e) {
    return fn.isFrontFacing(this.a, this.b, this.c, e);
  }
  /**
   * Returns `true` if this triangle intersects with the given box.
   *
   * @param {Box3} box - The box to intersect.
   * @return {boolean} Whether this triangle intersects with the given box or not.
   */
  intersectsBox(e) {
    return e.intersectsTriangle(this);
  }
  /**
   * Returns the closest point on the triangle to the given point.
   *
   * @param {Vector3} p - The point to compute the closest point for.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The closest point on the triangle.
   */
  closestPointToPoint(e, t) {
    const i = this.a, s = this.b, r = this.c;
    let o, a;
    fs.subVectors(s, i), ds.subVectors(r, i), La.subVectors(e, i);
    const l = fs.dot(La), c = ds.dot(La);
    if (l <= 0 && c <= 0)
      return t.copy(i);
    Ia.subVectors(e, s);
    const u = fs.dot(Ia), h = ds.dot(Ia);
    if (u >= 0 && h <= u)
      return t.copy(s);
    const f = l * h - u * c;
    if (f <= 0 && l >= 0 && u <= 0)
      return o = l / (l - u), t.copy(i).addScaledVector(fs, o);
    Ua.subVectors(e, r);
    const p = fs.dot(Ua), v = ds.dot(Ua);
    if (v >= 0 && p <= v)
      return t.copy(r);
    const x = p * c - l * v;
    if (x <= 0 && c >= 0 && v <= 0)
      return a = c / (c - v), t.copy(i).addScaledVector(ds, a);
    const m = u * v - p * h;
    if (m <= 0 && h - u >= 0 && p - v >= 0)
      return Wu.subVectors(r, s), a = (h - u) / (h - u + (p - v)), t.copy(s).addScaledVector(Wu, a);
    const d = 1 / (m + x + f);
    return o = x * d, a = f * d, t.copy(i).addScaledVector(fs, o).addScaledVector(ds, a);
  }
  /**
   * Returns `true` if this triangle is equal with the given one.
   *
   * @param {Triangle} triangle - The triangle to test for equality.
   * @return {boolean} Whether this triangle is equal with the given one.
   */
  equals(e) {
    return e.a.equals(this.a) && e.b.equals(this.b) && e.c.equals(this.c);
  }
}
const gd = {
  aliceblue: 15792383,
  antiquewhite: 16444375,
  aqua: 65535,
  aquamarine: 8388564,
  azure: 15794175,
  beige: 16119260,
  bisque: 16770244,
  black: 0,
  blanchedalmond: 16772045,
  blue: 255,
  blueviolet: 9055202,
  brown: 10824234,
  burlywood: 14596231,
  cadetblue: 6266528,
  chartreuse: 8388352,
  chocolate: 13789470,
  coral: 16744272,
  cornflowerblue: 6591981,
  cornsilk: 16775388,
  crimson: 14423100,
  cyan: 65535,
  darkblue: 139,
  darkcyan: 35723,
  darkgoldenrod: 12092939,
  darkgray: 11119017,
  darkgreen: 25600,
  darkgrey: 11119017,
  darkkhaki: 12433259,
  darkmagenta: 9109643,
  darkolivegreen: 5597999,
  darkorange: 16747520,
  darkorchid: 10040012,
  darkred: 9109504,
  darksalmon: 15308410,
  darkseagreen: 9419919,
  darkslateblue: 4734347,
  darkslategray: 3100495,
  darkslategrey: 3100495,
  darkturquoise: 52945,
  darkviolet: 9699539,
  deeppink: 16716947,
  deepskyblue: 49151,
  dimgray: 6908265,
  dimgrey: 6908265,
  dodgerblue: 2003199,
  firebrick: 11674146,
  floralwhite: 16775920,
  forestgreen: 2263842,
  fuchsia: 16711935,
  gainsboro: 14474460,
  ghostwhite: 16316671,
  gold: 16766720,
  goldenrod: 14329120,
  gray: 8421504,
  green: 32768,
  greenyellow: 11403055,
  grey: 8421504,
  honeydew: 15794160,
  hotpink: 16738740,
  indianred: 13458524,
  indigo: 4915330,
  ivory: 16777200,
  khaki: 15787660,
  lavender: 15132410,
  lavenderblush: 16773365,
  lawngreen: 8190976,
  lemonchiffon: 16775885,
  lightblue: 11393254,
  lightcoral: 15761536,
  lightcyan: 14745599,
  lightgoldenrodyellow: 16448210,
  lightgray: 13882323,
  lightgreen: 9498256,
  lightgrey: 13882323,
  lightpink: 16758465,
  lightsalmon: 16752762,
  lightseagreen: 2142890,
  lightskyblue: 8900346,
  lightslategray: 7833753,
  lightslategrey: 7833753,
  lightsteelblue: 11584734,
  lightyellow: 16777184,
  lime: 65280,
  limegreen: 3329330,
  linen: 16445670,
  magenta: 16711935,
  maroon: 8388608,
  mediumaquamarine: 6737322,
  mediumblue: 205,
  mediumorchid: 12211667,
  mediumpurple: 9662683,
  mediumseagreen: 3978097,
  mediumslateblue: 8087790,
  mediumspringgreen: 64154,
  mediumturquoise: 4772300,
  mediumvioletred: 13047173,
  midnightblue: 1644912,
  mintcream: 16121850,
  mistyrose: 16770273,
  moccasin: 16770229,
  navajowhite: 16768685,
  navy: 128,
  oldlace: 16643558,
  olive: 8421376,
  olivedrab: 7048739,
  orange: 16753920,
  orangered: 16729344,
  orchid: 14315734,
  palegoldenrod: 15657130,
  palegreen: 10025880,
  paleturquoise: 11529966,
  palevioletred: 14381203,
  papayawhip: 16773077,
  peachpuff: 16767673,
  peru: 13468991,
  pink: 16761035,
  plum: 14524637,
  powderblue: 11591910,
  purple: 8388736,
  rebeccapurple: 6697881,
  red: 16711680,
  rosybrown: 12357519,
  royalblue: 4286945,
  saddlebrown: 9127187,
  salmon: 16416882,
  sandybrown: 16032864,
  seagreen: 3050327,
  seashell: 16774638,
  sienna: 10506797,
  silver: 12632256,
  skyblue: 8900331,
  slateblue: 6970061,
  slategray: 7372944,
  slategrey: 7372944,
  snow: 16775930,
  springgreen: 65407,
  steelblue: 4620980,
  tan: 13808780,
  teal: 32896,
  thistle: 14204888,
  tomato: 16737095,
  turquoise: 4251856,
  violet: 15631086,
  wheat: 16113331,
  white: 16777215,
  whitesmoke: 16119285,
  yellow: 16776960,
  yellowgreen: 10145074
}, hi = { h: 0, s: 0, l: 0 }, $r = { h: 0, s: 0, l: 0 };
function Ba(n, e, t) {
  return t < 0 && (t += 1), t > 1 && (t -= 1), t < 1 / 6 ? n + (e - n) * 6 * t : t < 1 / 2 ? e : t < 2 / 3 ? n + (e - n) * 6 * (2 / 3 - t) : n;
}
class Xe {
  /**
   * Constructs a new color.
   *
   * Note that standard method of specifying color in three.js is with a hexadecimal triplet,
   * and that method is used throughout the rest of the documentation.
   *
   * @param {(number|string|Color)} [r] - The red component of the color. If `g` and `b` are
   * not provided, it can be hexadecimal triplet, a CSS-style string or another `Color` instance.
   * @param {number} [g] - The green component.
   * @param {number} [b] - The blue component.
   */
  constructor(e, t, i) {
    return this.isColor = !0, this.r = 1, this.g = 1, this.b = 1, this.set(e, t, i);
  }
  /**
   * Sets the colors's components from the given values.
   *
   * @param {(number|string|Color)} [r] - The red component of the color. If `g` and `b` are
   * not provided, it can be hexadecimal triplet, a CSS-style string or another `Color` instance.
   * @param {number} [g] - The green component.
   * @param {number} [b] - The blue component.
   * @return {Color} A reference to this color.
   */
  set(e, t, i) {
    if (t === void 0 && i === void 0) {
      const s = e;
      s && s.isColor ? this.copy(s) : typeof s == "number" ? this.setHex(s) : typeof s == "string" && this.setStyle(s);
    } else
      this.setRGB(e, t, i);
    return this;
  }
  /**
   * Sets the colors's components to the given scalar value.
   *
   * @param {number} scalar - The scalar value.
   * @return {Color} A reference to this color.
   */
  setScalar(e) {
    return this.r = e, this.g = e, this.b = e, this;
  }
  /**
   * Sets this color from a hexadecimal value.
   *
   * @param {number} hex - The hexadecimal value.
   * @param {string} [colorSpace=SRGBColorSpace] - The color space.
   * @return {Color} A reference to this color.
   */
  setHex(e, t = sn) {
    return e = Math.floor(e), this.r = (e >> 16 & 255) / 255, this.g = (e >> 8 & 255) / 255, this.b = (e & 255) / 255, et.colorSpaceToWorking(this, t), this;
  }
  /**
   * Sets this color from RGB values.
   *
   * @param {number} r - Red channel value between `0.0` and `1.0`.
   * @param {number} g - Green channel value between `0.0` and `1.0`.
   * @param {number} b - Blue channel value between `0.0` and `1.0`.
   * @param {string} [colorSpace=ColorManagement.workingColorSpace] - The color space.
   * @return {Color} A reference to this color.
   */
  setRGB(e, t, i, s = et.workingColorSpace) {
    return this.r = e, this.g = t, this.b = i, et.colorSpaceToWorking(this, s), this;
  }
  /**
   * Sets this color from RGB values.
   *
   * @param {number} h - Hue value between `0.0` and `1.0`.
   * @param {number} s - Saturation value between `0.0` and `1.0`.
   * @param {number} l - Lightness value between `0.0` and `1.0`.
   * @param {string} [colorSpace=ColorManagement.workingColorSpace] - The color space.
   * @return {Color} A reference to this color.
   */
  setHSL(e, t, i, s = et.workingColorSpace) {
    if (e = Dg(e, 1), t = Ke(t, 0, 1), i = Ke(i, 0, 1), t === 0)
      this.r = this.g = this.b = i;
    else {
      const r = i <= 0.5 ? i * (1 + t) : i + t - i * t, o = 2 * i - r;
      this.r = Ba(o, r, e + 1 / 3), this.g = Ba(o, r, e), this.b = Ba(o, r, e - 1 / 3);
    }
    return et.colorSpaceToWorking(this, s), this;
  }
  /**
   * Sets this color from a CSS-style string. For example, `rgb(250, 0,0)`,
   * `rgb(100%, 0%, 0%)`, `hsl(0, 100%, 50%)`, `#ff0000`, `#f00`, or `red` ( or
   * any [X11 color name]{@link https://en.wikipedia.org/wiki/X11_color_names#Color_name_chart} -
   * all 140 color names are supported).
   *
   * @param {string} style - Color as a CSS-style string.
   * @param {string} [colorSpace=SRGBColorSpace] - The color space.
   * @return {Color} A reference to this color.
   */
  setStyle(e, t = sn) {
    function i(r) {
      r !== void 0 && parseFloat(r) < 1 && console.warn("THREE.Color: Alpha component of " + e + " will be ignored.");
    }
    let s;
    if (s = /^(\w+)\(([^\)]*)\)/.exec(e)) {
      let r;
      const o = s[1], a = s[2];
      switch (o) {
        case "rgb":
        case "rgba":
          if (r = /^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))
            return i(r[4]), this.setRGB(
              Math.min(255, parseInt(r[1], 10)) / 255,
              Math.min(255, parseInt(r[2], 10)) / 255,
              Math.min(255, parseInt(r[3], 10)) / 255,
              t
            );
          if (r = /^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))
            return i(r[4]), this.setRGB(
              Math.min(100, parseInt(r[1], 10)) / 100,
              Math.min(100, parseInt(r[2], 10)) / 100,
              Math.min(100, parseInt(r[3], 10)) / 100,
              t
            );
          break;
        case "hsl":
        case "hsla":
          if (r = /^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))
            return i(r[4]), this.setHSL(
              parseFloat(r[1]) / 360,
              parseFloat(r[2]) / 100,
              parseFloat(r[3]) / 100,
              t
            );
          break;
        default:
          console.warn("THREE.Color: Unknown color model " + e);
      }
    } else if (s = /^\#([A-Fa-f\d]+)$/.exec(e)) {
      const r = s[1], o = r.length;
      if (o === 3)
        return this.setRGB(
          parseInt(r.charAt(0), 16) / 15,
          parseInt(r.charAt(1), 16) / 15,
          parseInt(r.charAt(2), 16) / 15,
          t
        );
      if (o === 6)
        return this.setHex(parseInt(r, 16), t);
      console.warn("THREE.Color: Invalid hex color " + e);
    } else if (e && e.length > 0)
      return this.setColorName(e, t);
    return this;
  }
  /**
   * Sets this color from a color name. Faster than {@link Color#setStyle} if
   * you don't need the other CSS-style formats.
   *
   * For convenience, the list of names is exposed in `Color.NAMES` as a hash.
   * ```js
   * Color.NAMES.aliceblue // returns 0xF0F8FF
   * ```
   *
   * @param {string} style - The color name.
   * @param {string} [colorSpace=SRGBColorSpace] - The color space.
   * @return {Color} A reference to this color.
   */
  setColorName(e, t = sn) {
    const i = gd[e.toLowerCase()];
    return i !== void 0 ? this.setHex(i, t) : console.warn("THREE.Color: Unknown color " + e), this;
  }
  /**
   * Returns a new color with copied values from this instance.
   *
   * @return {Color} A clone of this instance.
   */
  clone() {
    return new this.constructor(this.r, this.g, this.b);
  }
  /**
   * Copies the values of the given color to this instance.
   *
   * @param {Color} color - The color to copy.
   * @return {Color} A reference to this color.
   */
  copy(e) {
    return this.r = e.r, this.g = e.g, this.b = e.b, this;
  }
  /**
   * Copies the given color into this color, and then converts this color from
   * `SRGBColorSpace` to `LinearSRGBColorSpace`.
   *
   * @param {Color} color - The color to copy/convert.
   * @return {Color} A reference to this color.
   */
  copySRGBToLinear(e) {
    return this.r = ti(e.r), this.g = ti(e.g), this.b = ti(e.b), this;
  }
  /**
   * Copies the given color into this color, and then converts this color from
   * `LinearSRGBColorSpace` to `SRGBColorSpace`.
   *
   * @param {Color} color - The color to copy/convert.
   * @return {Color} A reference to this color.
   */
  copyLinearToSRGB(e) {
    return this.r = Is(e.r), this.g = Is(e.g), this.b = Is(e.b), this;
  }
  /**
   * Converts this color from `SRGBColorSpace` to `LinearSRGBColorSpace`.
   *
   * @return {Color} A reference to this color.
   */
  convertSRGBToLinear() {
    return this.copySRGBToLinear(this), this;
  }
  /**
   * Converts this color from `LinearSRGBColorSpace` to `SRGBColorSpace`.
   *
   * @return {Color} A reference to this color.
   */
  convertLinearToSRGB() {
    return this.copyLinearToSRGB(this), this;
  }
  /**
   * Returns the hexadecimal value of this color.
   *
   * @param {string} [colorSpace=SRGBColorSpace] - The color space.
   * @return {number} The hexadecimal value.
   */
  getHex(e = sn) {
    return et.workingToColorSpace(Dt.copy(this), e), Math.round(Ke(Dt.r * 255, 0, 255)) * 65536 + Math.round(Ke(Dt.g * 255, 0, 255)) * 256 + Math.round(Ke(Dt.b * 255, 0, 255));
  }
  /**
   * Returns the hexadecimal value of this color as a string (for example, 'FFFFFF').
   *
   * @param {string} [colorSpace=SRGBColorSpace] - The color space.
   * @return {string} The hexadecimal value as a string.
   */
  getHexString(e = sn) {
    return ("000000" + this.getHex(e).toString(16)).slice(-6);
  }
  /**
   * Converts the colors RGB values into the HSL format and stores them into the
   * given target object.
   *
   * @param {{h:number,s:number,l:number}} target - The target object that is used to store the method's result.
   * @param {string} [colorSpace=ColorManagement.workingColorSpace] - The color space.
   * @return {{h:number,s:number,l:number}} The HSL representation of this color.
   */
  getHSL(e, t = et.workingColorSpace) {
    et.workingToColorSpace(Dt.copy(this), t);
    const i = Dt.r, s = Dt.g, r = Dt.b, o = Math.max(i, s, r), a = Math.min(i, s, r);
    let l, c;
    const u = (a + o) / 2;
    if (a === o)
      l = 0, c = 0;
    else {
      const h = o - a;
      switch (c = u <= 0.5 ? h / (o + a) : h / (2 - o - a), o) {
        case i:
          l = (s - r) / h + (s < r ? 6 : 0);
          break;
        case s:
          l = (r - i) / h + 2;
          break;
        case r:
          l = (i - s) / h + 4;
          break;
      }
      l /= 6;
    }
    return e.h = l, e.s = c, e.l = u, e;
  }
  /**
   * Returns the RGB values of this color and stores them into the given target object.
   *
   * @param {Color} target - The target color that is used to store the method's result.
   * @param {string} [colorSpace=ColorManagement.workingColorSpace] - The color space.
   * @return {Color} The RGB representation of this color.
   */
  getRGB(e, t = et.workingColorSpace) {
    return et.workingToColorSpace(Dt.copy(this), t), e.r = Dt.r, e.g = Dt.g, e.b = Dt.b, e;
  }
  /**
   * Returns the value of this color as a CSS style string. Example: `rgb(255,0,0)`.
   *
   * @param {string} [colorSpace=SRGBColorSpace] - The color space.
   * @return {string} The CSS representation of this color.
   */
  getStyle(e = sn) {
    et.workingToColorSpace(Dt.copy(this), e);
    const t = Dt.r, i = Dt.g, s = Dt.b;
    return e !== sn ? `color(${e} ${t.toFixed(3)} ${i.toFixed(3)} ${s.toFixed(3)})` : `rgb(${Math.round(t * 255)},${Math.round(i * 255)},${Math.round(s * 255)})`;
  }
  /**
   * Adds the given HSL values to this color's values.
   * Internally, this converts the color's RGB values to HSL, adds HSL
   * and then converts the color back to RGB.
   *
   * @param {number} h - Hue value between `0.0` and `1.0`.
   * @param {number} s - Saturation value between `0.0` and `1.0`.
   * @param {number} l - Lightness value between `0.0` and `1.0`.
   * @return {Color} A reference to this color.
   */
  offsetHSL(e, t, i) {
    return this.getHSL(hi), this.setHSL(hi.h + e, hi.s + t, hi.l + i);
  }
  /**
   * Adds the RGB values of the given color to the RGB values of this color.
   *
   * @param {Color} color - The color to add.
   * @return {Color} A reference to this color.
   */
  add(e) {
    return this.r += e.r, this.g += e.g, this.b += e.b, this;
  }
  /**
   * Adds the RGB values of the given colors and stores the result in this instance.
   *
   * @param {Color} color1 - The first color.
   * @param {Color} color2 - The second color.
   * @return {Color} A reference to this color.
   */
  addColors(e, t) {
    return this.r = e.r + t.r, this.g = e.g + t.g, this.b = e.b + t.b, this;
  }
  /**
   * Adds the given scalar value to the RGB values of this color.
   *
   * @param {number} s - The scalar to add.
   * @return {Color} A reference to this color.
   */
  addScalar(e) {
    return this.r += e, this.g += e, this.b += e, this;
  }
  /**
   * Subtracts the RGB values of the given color from the RGB values of this color.
   *
   * @param {Color} color - The color to subtract.
   * @return {Color} A reference to this color.
   */
  sub(e) {
    return this.r = Math.max(0, this.r - e.r), this.g = Math.max(0, this.g - e.g), this.b = Math.max(0, this.b - e.b), this;
  }
  /**
   * Multiplies the RGB values of the given color with the RGB values of this color.
   *
   * @param {Color} color - The color to multiply.
   * @return {Color} A reference to this color.
   */
  multiply(e) {
    return this.r *= e.r, this.g *= e.g, this.b *= e.b, this;
  }
  /**
   * Multiplies the given scalar value with the RGB values of this color.
   *
   * @param {number} s - The scalar to multiply.
   * @return {Color} A reference to this color.
   */
  multiplyScalar(e) {
    return this.r *= e, this.g *= e, this.b *= e, this;
  }
  /**
   * Linearly interpolates this color's RGB values toward the RGB values of the
   * given color. The alpha argument can be thought of as the ratio between
   * the two colors, where `0.0` is this color and `1.0` is the first argument.
   *
   * @param {Color} color - The color to converge on.
   * @param {number} alpha - The interpolation factor in the closed interval `[0,1]`.
   * @return {Color} A reference to this color.
   */
  lerp(e, t) {
    return this.r += (e.r - this.r) * t, this.g += (e.g - this.g) * t, this.b += (e.b - this.b) * t, this;
  }
  /**
   * Linearly interpolates between the given colors and stores the result in this instance.
   * The alpha argument can be thought of as the ratio between the two colors, where `0.0`
   * is the first and `1.0` is the second color.
   *
   * @param {Color} color1 - The first color.
   * @param {Color} color2 - The second color.
   * @param {number} alpha - The interpolation factor in the closed interval `[0,1]`.
   * @return {Color} A reference to this color.
   */
  lerpColors(e, t, i) {
    return this.r = e.r + (t.r - e.r) * i, this.g = e.g + (t.g - e.g) * i, this.b = e.b + (t.b - e.b) * i, this;
  }
  /**
   * Linearly interpolates this color's HSL values toward the HSL values of the
   * given color. It differs from {@link Color#lerp} by not interpolating straight
   * from one color to the other, but instead going through all the hues in between
   * those two colors. The alpha argument can be thought of as the ratio between
   * the two colors, where 0.0 is this color and 1.0 is the first argument.
   *
   * @param {Color} color - The color to converge on.
   * @param {number} alpha - The interpolation factor in the closed interval `[0,1]`.
   * @return {Color} A reference to this color.
   */
  lerpHSL(e, t) {
    this.getHSL(hi), e.getHSL($r);
    const i = Ma(hi.h, $r.h, t), s = Ma(hi.s, $r.s, t), r = Ma(hi.l, $r.l, t);
    return this.setHSL(i, s, r), this;
  }
  /**
   * Sets the color's RGB components from the given 3D vector.
   *
   * @param {Vector3} v - The vector to set.
   * @return {Color} A reference to this color.
   */
  setFromVector3(e) {
    return this.r = e.x, this.g = e.y, this.b = e.z, this;
  }
  /**
   * Transforms this color with the given 3x3 matrix.
   *
   * @param {Matrix3} m - The matrix.
   * @return {Color} A reference to this color.
   */
  applyMatrix3(e) {
    const t = this.r, i = this.g, s = this.b, r = e.elements;
    return this.r = r[0] * t + r[3] * i + r[6] * s, this.g = r[1] * t + r[4] * i + r[7] * s, this.b = r[2] * t + r[5] * i + r[8] * s, this;
  }
  /**
   * Returns `true` if this color is equal with the given one.
   *
   * @param {Color} c - The color to test for equality.
   * @return {boolean} Whether this bounding color is equal with the given one.
   */
  equals(e) {
    return e.r === this.r && e.g === this.g && e.b === this.b;
  }
  /**
   * Sets this color's RGB components from the given array.
   *
   * @param {Array<number>} array - An array holding the RGB values.
   * @param {number} [offset=0] - The offset into the array.
   * @return {Color} A reference to this color.
   */
  fromArray(e, t = 0) {
    return this.r = e[t], this.g = e[t + 1], this.b = e[t + 2], this;
  }
  /**
   * Writes the RGB components of this color to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the color components.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The color components.
   */
  toArray(e = [], t = 0) {
    return e[t] = this.r, e[t + 1] = this.g, e[t + 2] = this.b, e;
  }
  /**
   * Sets the components of this color from the given buffer attribute.
   *
   * @param {BufferAttribute} attribute - The buffer attribute holding color data.
   * @param {number} index - The index into the attribute.
   * @return {Color} A reference to this color.
   */
  fromBufferAttribute(e, t) {
    return this.r = e.getX(t), this.g = e.getY(t), this.b = e.getZ(t), this;
  }
  /**
   * This methods defines the serialization result of this class. Returns the color
   * as a hexadecimal value.
   *
   * @return {number} The hexadecimal value.
   */
  toJSON() {
    return this.getHex();
  }
  *[Symbol.iterator]() {
    yield this.r, yield this.g, yield this.b;
  }
}
const Dt = /* @__PURE__ */ new Xe();
Xe.NAMES = gd;
let jg = 0;
class Qi extends Ji {
  /**
   * Constructs a new material.
   */
  constructor() {
    super(), this.isMaterial = !0, Object.defineProperty(this, "id", { value: jg++ }), this.uuid = Ur(), this.name = "", this.type = "Material", this.blending = Ls, this.side = yi, this.vertexColors = !1, this.opacity = 1, this.transparent = !1, this.alphaHash = !1, this.blendSrc = fl, this.blendDst = dl, this.blendEquation = zi, this.blendSrcAlpha = null, this.blendDstAlpha = null, this.blendEquationAlpha = null, this.blendColor = new Xe(0, 0, 0), this.blendAlpha = 0, this.depthFunc = Ns, this.depthTest = !0, this.depthWrite = !0, this.stencilWriteMask = 255, this.stencilFunc = Pu, this.stencilRef = 0, this.stencilFuncMask = 255, this.stencilFail = ss, this.stencilZFail = ss, this.stencilZPass = ss, this.stencilWrite = !1, this.clippingPlanes = null, this.clipIntersection = !1, this.clipShadows = !1, this.shadowSide = null, this.colorWrite = !0, this.precision = null, this.polygonOffset = !1, this.polygonOffsetFactor = 0, this.polygonOffsetUnits = 0, this.dithering = !1, this.alphaToCoverage = !1, this.premultipliedAlpha = !1, this.forceSinglePass = !1, this.allowOverride = !0, this.visible = !0, this.toneMapped = !0, this.userData = {}, this.version = 0, this._alphaTest = 0;
  }
  /**
   * Sets the alpha value to be used when running an alpha test. The material
   * will not be rendered if the opacity is lower than this value.
   *
   * @type {number}
   * @readonly
   * @default 0
   */
  get alphaTest() {
    return this._alphaTest;
  }
  set alphaTest(e) {
    this._alphaTest > 0 != e > 0 && this.version++, this._alphaTest = e;
  }
  /**
   * An optional callback that is executed immediately before the material is used to render a 3D object.
   *
   * This method can only be used when rendering with {@link WebGLRenderer}.
   *
   * @param {WebGLRenderer} renderer - The renderer.
   * @param {Scene} scene - The scene.
   * @param {Camera} camera - The camera that is used to render the scene.
   * @param {BufferGeometry} geometry - The 3D object's geometry.
   * @param {Object3D} object - The 3D object.
   * @param {Object} group - The geometry group data.
   */
  onBeforeRender() {
  }
  /**
   * An optional callback that is executed immediately before the shader
   * program is compiled. This function is called with the shader source code
   * as a parameter. Useful for the modification of built-in materials.
   *
   * This method can only be used when rendering with {@link WebGLRenderer}. The
   * recommended approach when customizing materials is to use `WebGPURenderer` with the new
   * Node Material system and [TSL]{@link https://github.com/mrdoob/three.js/wiki/Three.js-Shading-Language}.
   *
   * @param {{vertexShader:string,fragmentShader:string,uniforms:Object}} shaderobject - The object holds the uniforms and the vertex and fragment shader source.
   * @param {WebGLRenderer} renderer - A reference to the renderer.
   */
  onBeforeCompile() {
  }
  /**
   * In case {@link Material#onBeforeCompile} is used, this callback can be used to identify
   * values of settings used in `onBeforeCompile()`, so three.js can reuse a cached
   * shader or recompile the shader for this material as needed.
   *
   * This method can only be used when rendering with {@link WebGLRenderer}.
   *
   * @return {string} The custom program cache key.
   */
  customProgramCacheKey() {
    return this.onBeforeCompile.toString();
  }
  /**
   * This method can be used to set default values from parameter objects.
   * It is a generic implementation so it can be used with different types
   * of materials.
   *
   * @param {Object} [values] - The material values to set.
   */
  setValues(e) {
    if (e !== void 0)
      for (const t in e) {
        const i = e[t];
        if (i === void 0) {
          console.warn(`THREE.Material: parameter '${t}' has value of undefined.`);
          continue;
        }
        const s = this[t];
        if (s === void 0) {
          console.warn(`THREE.Material: '${t}' is not a property of THREE.${this.type}.`);
          continue;
        }
        s && s.isColor ? s.set(i) : s && s.isVector3 && i && i.isVector3 ? s.copy(i) : this[t] = i;
      }
  }
  /**
   * Serializes the material into JSON.
   *
   * @param {?(Object|string)} meta - An optional value holding meta information about the serialization.
   * @return {Object} A JSON object representing the serialized material.
   * @see {@link ObjectLoader#parse}
   */
  toJSON(e) {
    const t = e === void 0 || typeof e == "string";
    t && (e = {
      textures: {},
      images: {}
    });
    const i = {
      metadata: {
        version: 4.7,
        type: "Material",
        generator: "Material.toJSON"
      }
    };
    i.uuid = this.uuid, i.type = this.type, this.name !== "" && (i.name = this.name), this.color && this.color.isColor && (i.color = this.color.getHex()), this.roughness !== void 0 && (i.roughness = this.roughness), this.metalness !== void 0 && (i.metalness = this.metalness), this.sheen !== void 0 && (i.sheen = this.sheen), this.sheenColor && this.sheenColor.isColor && (i.sheenColor = this.sheenColor.getHex()), this.sheenRoughness !== void 0 && (i.sheenRoughness = this.sheenRoughness), this.emissive && this.emissive.isColor && (i.emissive = this.emissive.getHex()), this.emissiveIntensity !== void 0 && this.emissiveIntensity !== 1 && (i.emissiveIntensity = this.emissiveIntensity), this.specular && this.specular.isColor && (i.specular = this.specular.getHex()), this.specularIntensity !== void 0 && (i.specularIntensity = this.specularIntensity), this.specularColor && this.specularColor.isColor && (i.specularColor = this.specularColor.getHex()), this.shininess !== void 0 && (i.shininess = this.shininess), this.clearcoat !== void 0 && (i.clearcoat = this.clearcoat), this.clearcoatRoughness !== void 0 && (i.clearcoatRoughness = this.clearcoatRoughness), this.clearcoatMap && this.clearcoatMap.isTexture && (i.clearcoatMap = this.clearcoatMap.toJSON(e).uuid), this.clearcoatRoughnessMap && this.clearcoatRoughnessMap.isTexture && (i.clearcoatRoughnessMap = this.clearcoatRoughnessMap.toJSON(e).uuid), this.clearcoatNormalMap && this.clearcoatNormalMap.isTexture && (i.clearcoatNormalMap = this.clearcoatNormalMap.toJSON(e).uuid, i.clearcoatNormalScale = this.clearcoatNormalScale.toArray()), this.sheenColorMap && this.sheenColorMap.isTexture && (i.sheenColorMap = this.sheenColorMap.toJSON(e).uuid), this.sheenRoughnessMap && this.sheenRoughnessMap.isTexture && (i.sheenRoughnessMap = this.sheenRoughnessMap.toJSON(e).uuid), this.dispersion !== void 0 && (i.dispersion = this.dispersion), this.iridescence !== void 0 && (i.iridescence = this.iridescence), this.iridescenceIOR !== void 0 && (i.iridescenceIOR = this.iridescenceIOR), this.iridescenceThicknessRange !== void 0 && (i.iridescenceThicknessRange = this.iridescenceThicknessRange), this.iridescenceMap && this.iridescenceMap.isTexture && (i.iridescenceMap = this.iridescenceMap.toJSON(e).uuid), this.iridescenceThicknessMap && this.iridescenceThicknessMap.isTexture && (i.iridescenceThicknessMap = this.iridescenceThicknessMap.toJSON(e).uuid), this.anisotropy !== void 0 && (i.anisotropy = this.anisotropy), this.anisotropyRotation !== void 0 && (i.anisotropyRotation = this.anisotropyRotation), this.anisotropyMap && this.anisotropyMap.isTexture && (i.anisotropyMap = this.anisotropyMap.toJSON(e).uuid), this.map && this.map.isTexture && (i.map = this.map.toJSON(e).uuid), this.matcap && this.matcap.isTexture && (i.matcap = this.matcap.toJSON(e).uuid), this.alphaMap && this.alphaMap.isTexture && (i.alphaMap = this.alphaMap.toJSON(e).uuid), this.lightMap && this.lightMap.isTexture && (i.lightMap = this.lightMap.toJSON(e).uuid, i.lightMapIntensity = this.lightMapIntensity), this.aoMap && this.aoMap.isTexture && (i.aoMap = this.aoMap.toJSON(e).uuid, i.aoMapIntensity = this.aoMapIntensity), this.bumpMap && this.bumpMap.isTexture && (i.bumpMap = this.bumpMap.toJSON(e).uuid, i.bumpScale = this.bumpScale), this.normalMap && this.normalMap.isTexture && (i.normalMap = this.normalMap.toJSON(e).uuid, i.normalMapType = this.normalMapType, i.normalScale = this.normalScale.toArray()), this.displacementMap && this.displacementMap.isTexture && (i.displacementMap = this.displacementMap.toJSON(e).uuid, i.displacementScale = this.displacementScale, i.displacementBias = this.displacementBias), this.roughnessMap && this.roughnessMap.isTexture && (i.roughnessMap = this.roughnessMap.toJSON(e).uuid), this.metalnessMap && this.metalnessMap.isTexture && (i.metalnessMap = this.metalnessMap.toJSON(e).uuid), this.emissiveMap && this.emissiveMap.isTexture && (i.emissiveMap = this.emissiveMap.toJSON(e).uuid), this.specularMap && this.specularMap.isTexture && (i.specularMap = this.specularMap.toJSON(e).uuid), this.specularIntensityMap && this.specularIntensityMap.isTexture && (i.specularIntensityMap = this.specularIntensityMap.toJSON(e).uuid), this.specularColorMap && this.specularColorMap.isTexture && (i.specularColorMap = this.specularColorMap.toJSON(e).uuid), this.envMap && this.envMap.isTexture && (i.envMap = this.envMap.toJSON(e).uuid, this.combine !== void 0 && (i.combine = this.combine)), this.envMapRotation !== void 0 && (i.envMapRotation = this.envMapRotation.toArray()), this.envMapIntensity !== void 0 && (i.envMapIntensity = this.envMapIntensity), this.reflectivity !== void 0 && (i.reflectivity = this.reflectivity), this.refractionRatio !== void 0 && (i.refractionRatio = this.refractionRatio), this.gradientMap && this.gradientMap.isTexture && (i.gradientMap = this.gradientMap.toJSON(e).uuid), this.transmission !== void 0 && (i.transmission = this.transmission), this.transmissionMap && this.transmissionMap.isTexture && (i.transmissionMap = this.transmissionMap.toJSON(e).uuid), this.thickness !== void 0 && (i.thickness = this.thickness), this.thicknessMap && this.thicknessMap.isTexture && (i.thicknessMap = this.thicknessMap.toJSON(e).uuid), this.attenuationDistance !== void 0 && this.attenuationDistance !== 1 / 0 && (i.attenuationDistance = this.attenuationDistance), this.attenuationColor !== void 0 && (i.attenuationColor = this.attenuationColor.getHex()), this.size !== void 0 && (i.size = this.size), this.shadowSide !== null && (i.shadowSide = this.shadowSide), this.sizeAttenuation !== void 0 && (i.sizeAttenuation = this.sizeAttenuation), this.blending !== Ls && (i.blending = this.blending), this.side !== yi && (i.side = this.side), this.vertexColors === !0 && (i.vertexColors = !0), this.opacity < 1 && (i.opacity = this.opacity), this.transparent === !0 && (i.transparent = !0), this.blendSrc !== fl && (i.blendSrc = this.blendSrc), this.blendDst !== dl && (i.blendDst = this.blendDst), this.blendEquation !== zi && (i.blendEquation = this.blendEquation), this.blendSrcAlpha !== null && (i.blendSrcAlpha = this.blendSrcAlpha), this.blendDstAlpha !== null && (i.blendDstAlpha = this.blendDstAlpha), this.blendEquationAlpha !== null && (i.blendEquationAlpha = this.blendEquationAlpha), this.blendColor && this.blendColor.isColor && (i.blendColor = this.blendColor.getHex()), this.blendAlpha !== 0 && (i.blendAlpha = this.blendAlpha), this.depthFunc !== Ns && (i.depthFunc = this.depthFunc), this.depthTest === !1 && (i.depthTest = this.depthTest), this.depthWrite === !1 && (i.depthWrite = this.depthWrite), this.colorWrite === !1 && (i.colorWrite = this.colorWrite), this.stencilWriteMask !== 255 && (i.stencilWriteMask = this.stencilWriteMask), this.stencilFunc !== Pu && (i.stencilFunc = this.stencilFunc), this.stencilRef !== 0 && (i.stencilRef = this.stencilRef), this.stencilFuncMask !== 255 && (i.stencilFuncMask = this.stencilFuncMask), this.stencilFail !== ss && (i.stencilFail = this.stencilFail), this.stencilZFail !== ss && (i.stencilZFail = this.stencilZFail), this.stencilZPass !== ss && (i.stencilZPass = this.stencilZPass), this.stencilWrite === !0 && (i.stencilWrite = this.stencilWrite), this.rotation !== void 0 && this.rotation !== 0 && (i.rotation = this.rotation), this.polygonOffset === !0 && (i.polygonOffset = !0), this.polygonOffsetFactor !== 0 && (i.polygonOffsetFactor = this.polygonOffsetFactor), this.polygonOffsetUnits !== 0 && (i.polygonOffsetUnits = this.polygonOffsetUnits), this.linewidth !== void 0 && this.linewidth !== 1 && (i.linewidth = this.linewidth), this.dashSize !== void 0 && (i.dashSize = this.dashSize), this.gapSize !== void 0 && (i.gapSize = this.gapSize), this.scale !== void 0 && (i.scale = this.scale), this.dithering === !0 && (i.dithering = !0), this.alphaTest > 0 && (i.alphaTest = this.alphaTest), this.alphaHash === !0 && (i.alphaHash = !0), this.alphaToCoverage === !0 && (i.alphaToCoverage = !0), this.premultipliedAlpha === !0 && (i.premultipliedAlpha = !0), this.forceSinglePass === !0 && (i.forceSinglePass = !0), this.wireframe === !0 && (i.wireframe = !0), this.wireframeLinewidth > 1 && (i.wireframeLinewidth = this.wireframeLinewidth), this.wireframeLinecap !== "round" && (i.wireframeLinecap = this.wireframeLinecap), this.wireframeLinejoin !== "round" && (i.wireframeLinejoin = this.wireframeLinejoin), this.flatShading === !0 && (i.flatShading = !0), this.visible === !1 && (i.visible = !1), this.toneMapped === !1 && (i.toneMapped = !1), this.fog === !1 && (i.fog = !1), Object.keys(this.userData).length > 0 && (i.userData = this.userData);
    function s(r) {
      const o = [];
      for (const a in r) {
        const l = r[a];
        delete l.metadata, o.push(l);
      }
      return o;
    }
    if (t) {
      const r = s(e.textures), o = s(e.images);
      r.length > 0 && (i.textures = r), o.length > 0 && (i.images = o);
    }
    return i;
  }
  /**
   * Returns a new material with copied values from this instance.
   *
   * @return {Material} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Copies the values of the given material to this instance.
   *
   * @param {Material} source - The material to copy.
   * @return {Material} A reference to this instance.
   */
  copy(e) {
    this.name = e.name, this.blending = e.blending, this.side = e.side, this.vertexColors = e.vertexColors, this.opacity = e.opacity, this.transparent = e.transparent, this.blendSrc = e.blendSrc, this.blendDst = e.blendDst, this.blendEquation = e.blendEquation, this.blendSrcAlpha = e.blendSrcAlpha, this.blendDstAlpha = e.blendDstAlpha, this.blendEquationAlpha = e.blendEquationAlpha, this.blendColor.copy(e.blendColor), this.blendAlpha = e.blendAlpha, this.depthFunc = e.depthFunc, this.depthTest = e.depthTest, this.depthWrite = e.depthWrite, this.stencilWriteMask = e.stencilWriteMask, this.stencilFunc = e.stencilFunc, this.stencilRef = e.stencilRef, this.stencilFuncMask = e.stencilFuncMask, this.stencilFail = e.stencilFail, this.stencilZFail = e.stencilZFail, this.stencilZPass = e.stencilZPass, this.stencilWrite = e.stencilWrite;
    const t = e.clippingPlanes;
    let i = null;
    if (t !== null) {
      const s = t.length;
      i = new Array(s);
      for (let r = 0; r !== s; ++r)
        i[r] = t[r].clone();
    }
    return this.clippingPlanes = i, this.clipIntersection = e.clipIntersection, this.clipShadows = e.clipShadows, this.shadowSide = e.shadowSide, this.colorWrite = e.colorWrite, this.precision = e.precision, this.polygonOffset = e.polygonOffset, this.polygonOffsetFactor = e.polygonOffsetFactor, this.polygonOffsetUnits = e.polygonOffsetUnits, this.dithering = e.dithering, this.alphaTest = e.alphaTest, this.alphaHash = e.alphaHash, this.alphaToCoverage = e.alphaToCoverage, this.premultipliedAlpha = e.premultipliedAlpha, this.forceSinglePass = e.forceSinglePass, this.visible = e.visible, this.toneMapped = e.toneMapped, this.userData = JSON.parse(JSON.stringify(e.userData)), this;
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   *
   * @fires Material#dispose
   */
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
  /**
   * Setting this property to `true` indicates the engine the material
   * needs to be recompiled.
   *
   * @type {boolean}
   * @default false
   * @param {boolean} value
   */
  set needsUpdate(e) {
    e === !0 && this.version++;
  }
}
class Rn extends Qi {
  /**
   * Constructs a new mesh basic material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isMeshBasicMaterial = !0, this.type = "MeshBasicMaterial", this.color = new Xe(16777215), this.map = null, this.lightMap = null, this.lightMapIntensity = 1, this.aoMap = null, this.aoMapIntensity = 1, this.specularMap = null, this.alphaMap = null, this.envMap = null, this.envMapRotation = new zn(), this.combine = td, this.reflectivity = 1, this.refractionRatio = 0.98, this.wireframe = !1, this.wireframeLinewidth = 1, this.wireframeLinecap = "round", this.wireframeLinejoin = "round", this.fog = !0, this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.color.copy(e.color), this.map = e.map, this.lightMap = e.lightMap, this.lightMapIntensity = e.lightMapIntensity, this.aoMap = e.aoMap, this.aoMapIntensity = e.aoMapIntensity, this.specularMap = e.specularMap, this.alphaMap = e.alphaMap, this.envMap = e.envMap, this.envMapRotation.copy(e.envMapRotation), this.combine = e.combine, this.reflectivity = e.reflectivity, this.refractionRatio = e.refractionRatio, this.wireframe = e.wireframe, this.wireframeLinewidth = e.wireframeLinewidth, this.wireframeLinecap = e.wireframeLinecap, this.wireframeLinejoin = e.wireframeLinejoin, this.fog = e.fog, this;
  }
}
const St = /* @__PURE__ */ new N(), Zr = /* @__PURE__ */ new Ve();
let Kg = 0;
class En {
  /**
   * Constructs a new buffer attribute.
   *
   * @param {TypedArray} array - The array holding the attribute data.
   * @param {number} itemSize - The item size.
   * @param {boolean} [normalized=false] - Whether the data are normalized or not.
   */
  constructor(e, t, i = !1) {
    if (Array.isArray(e))
      throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");
    this.isBufferAttribute = !0, Object.defineProperty(this, "id", { value: Kg++ }), this.name = "", this.array = e, this.itemSize = t, this.count = e !== void 0 ? e.length / t : 0, this.normalized = i, this.usage = Du, this.updateRanges = [], this.gpuType = ei, this.version = 0;
  }
  /**
   * A callback function that is executed after the renderer has transferred the attribute
   * array data to the GPU.
   */
  onUploadCallback() {
  }
  /**
   * Flag to indicate that this attribute has changed and should be re-sent to
   * the GPU. Set this to `true` when you modify the value of the array.
   *
   * @type {number}
   * @default false
   * @param {boolean} value
   */
  set needsUpdate(e) {
    e === !0 && this.version++;
  }
  /**
   * Sets the usage of this buffer attribute.
   *
   * @param {(StaticDrawUsage|DynamicDrawUsage|StreamDrawUsage|StaticReadUsage|DynamicReadUsage|StreamReadUsage|StaticCopyUsage|DynamicCopyUsage|StreamCopyUsage)} value - The usage to set.
   * @return {BufferAttribute} A reference to this buffer attribute.
   */
  setUsage(e) {
    return this.usage = e, this;
  }
  /**
   * Adds a range of data in the data array to be updated on the GPU.
   *
   * @param {number} start - Position at which to start update.
   * @param {number} count - The number of components to update.
   */
  addUpdateRange(e, t) {
    this.updateRanges.push({ start: e, count: t });
  }
  /**
   * Clears the update ranges.
   */
  clearUpdateRanges() {
    this.updateRanges.length = 0;
  }
  /**
   * Copies the values of the given buffer attribute to this instance.
   *
   * @param {BufferAttribute} source - The buffer attribute to copy.
   * @return {BufferAttribute} A reference to this instance.
   */
  copy(e) {
    return this.name = e.name, this.array = new e.array.constructor(e.array), this.itemSize = e.itemSize, this.count = e.count, this.normalized = e.normalized, this.usage = e.usage, this.gpuType = e.gpuType, this;
  }
  /**
   * Copies a vector from the given buffer attribute to this one. The start
   * and destination position in the attribute buffers are represented by the
   * given indices.
   *
   * @param {number} index1 - The destination index into this buffer attribute.
   * @param {BufferAttribute} attribute - The buffer attribute to copy from.
   * @param {number} index2 - The source index into the given buffer attribute.
   * @return {BufferAttribute} A reference to this instance.
   */
  copyAt(e, t, i) {
    e *= this.itemSize, i *= t.itemSize;
    for (let s = 0, r = this.itemSize; s < r; s++)
      this.array[e + s] = t.array[i + s];
    return this;
  }
  /**
   * Copies the given array data into this buffer attribute.
   *
   * @param {(TypedArray|Array)} array - The array to copy.
   * @return {BufferAttribute} A reference to this instance.
   */
  copyArray(e) {
    return this.array.set(e), this;
  }
  /**
   * Applies the given 3x3 matrix to the given attribute. Works with
   * item size `2` and `3`.
   *
   * @param {Matrix3} m - The matrix to apply.
   * @return {BufferAttribute} A reference to this instance.
   */
  applyMatrix3(e) {
    if (this.itemSize === 2)
      for (let t = 0, i = this.count; t < i; t++)
        Zr.fromBufferAttribute(this, t), Zr.applyMatrix3(e), this.setXY(t, Zr.x, Zr.y);
    else if (this.itemSize === 3)
      for (let t = 0, i = this.count; t < i; t++)
        St.fromBufferAttribute(this, t), St.applyMatrix3(e), this.setXYZ(t, St.x, St.y, St.z);
    return this;
  }
  /**
   * Applies the given 4x4 matrix to the given attribute. Only works with
   * item size `3`.
   *
   * @param {Matrix4} m - The matrix to apply.
   * @return {BufferAttribute} A reference to this instance.
   */
  applyMatrix4(e) {
    for (let t = 0, i = this.count; t < i; t++)
      St.fromBufferAttribute(this, t), St.applyMatrix4(e), this.setXYZ(t, St.x, St.y, St.z);
    return this;
  }
  /**
   * Applies the given 3x3 normal matrix to the given attribute. Only works with
   * item size `3`.
   *
   * @param {Matrix3} m - The normal matrix to apply.
   * @return {BufferAttribute} A reference to this instance.
   */
  applyNormalMatrix(e) {
    for (let t = 0, i = this.count; t < i; t++)
      St.fromBufferAttribute(this, t), St.applyNormalMatrix(e), this.setXYZ(t, St.x, St.y, St.z);
    return this;
  }
  /**
   * Applies the given 4x4 matrix to the given attribute. Only works with
   * item size `3` and with direction vectors.
   *
   * @param {Matrix4} m - The matrix to apply.
   * @return {BufferAttribute} A reference to this instance.
   */
  transformDirection(e) {
    for (let t = 0, i = this.count; t < i; t++)
      St.fromBufferAttribute(this, t), St.transformDirection(e), this.setXYZ(t, St.x, St.y, St.z);
    return this;
  }
  /**
   * Sets the given array data in the buffer attribute.
   *
   * @param {(TypedArray|Array)} value - The array data to set.
   * @param {number} [offset=0] - The offset in this buffer attribute's array.
   * @return {BufferAttribute} A reference to this instance.
   */
  set(e, t = 0) {
    return this.array.set(e, t), this;
  }
  /**
   * Returns the given component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} component - The component index.
   * @return {number} The returned value.
   */
  getComponent(e, t) {
    let i = this.array[e * this.itemSize + t];
    return this.normalized && (i = Ks(i, this.array)), i;
  }
  /**
   * Sets the given value to the given component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} component - The component index.
   * @param {number} value - The value to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setComponent(e, t, i) {
    return this.normalized && (i = Yt(i, this.array)), this.array[e * this.itemSize + t] = i, this;
  }
  /**
   * Returns the x component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @return {number} The x component.
   */
  getX(e) {
    let t = this.array[e * this.itemSize];
    return this.normalized && (t = Ks(t, this.array)), t;
  }
  /**
   * Sets the x component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} x - The value to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setX(e, t) {
    return this.normalized && (t = Yt(t, this.array)), this.array[e * this.itemSize] = t, this;
  }
  /**
   * Returns the y component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @return {number} The y component.
   */
  getY(e) {
    let t = this.array[e * this.itemSize + 1];
    return this.normalized && (t = Ks(t, this.array)), t;
  }
  /**
   * Sets the y component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} y - The value to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setY(e, t) {
    return this.normalized && (t = Yt(t, this.array)), this.array[e * this.itemSize + 1] = t, this;
  }
  /**
   * Returns the z component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @return {number} The z component.
   */
  getZ(e) {
    let t = this.array[e * this.itemSize + 2];
    return this.normalized && (t = Ks(t, this.array)), t;
  }
  /**
   * Sets the z component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} z - The value to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setZ(e, t) {
    return this.normalized && (t = Yt(t, this.array)), this.array[e * this.itemSize + 2] = t, this;
  }
  /**
   * Returns the w component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @return {number} The w component.
   */
  getW(e) {
    let t = this.array[e * this.itemSize + 3];
    return this.normalized && (t = Ks(t, this.array)), t;
  }
  /**
   * Sets the w component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} w - The value to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setW(e, t) {
    return this.normalized && (t = Yt(t, this.array)), this.array[e * this.itemSize + 3] = t, this;
  }
  /**
   * Sets the x and y component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} x - The value for the x component to set.
   * @param {number} y - The value for the y component to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setXY(e, t, i) {
    return e *= this.itemSize, this.normalized && (t = Yt(t, this.array), i = Yt(i, this.array)), this.array[e + 0] = t, this.array[e + 1] = i, this;
  }
  /**
   * Sets the x, y and z component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} x - The value for the x component to set.
   * @param {number} y - The value for the y component to set.
   * @param {number} z - The value for the z component to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setXYZ(e, t, i, s) {
    return e *= this.itemSize, this.normalized && (t = Yt(t, this.array), i = Yt(i, this.array), s = Yt(s, this.array)), this.array[e + 0] = t, this.array[e + 1] = i, this.array[e + 2] = s, this;
  }
  /**
   * Sets the x, y, z and w component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} x - The value for the x component to set.
   * @param {number} y - The value for the y component to set.
   * @param {number} z - The value for the z component to set.
   * @param {number} w - The value for the w component to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setXYZW(e, t, i, s, r) {
    return e *= this.itemSize, this.normalized && (t = Yt(t, this.array), i = Yt(i, this.array), s = Yt(s, this.array), r = Yt(r, this.array)), this.array[e + 0] = t, this.array[e + 1] = i, this.array[e + 2] = s, this.array[e + 3] = r, this;
  }
  /**
   * Sets the given callback function that is executed after the Renderer has transferred
   * the attribute array data to the GPU. Can be used to perform clean-up operations after
   * the upload when attribute data are not needed anymore on the CPU side.
   *
   * @param {Function} callback - The `onUpload()` callback.
   * @return {BufferAttribute} A reference to this instance.
   */
  onUpload(e) {
    return this.onUploadCallback = e, this;
  }
  /**
   * Returns a new buffer attribute with copied values from this instance.
   *
   * @return {BufferAttribute} A clone of this instance.
   */
  clone() {
    return new this.constructor(this.array, this.itemSize).copy(this);
  }
  /**
   * Serializes the buffer attribute into JSON.
   *
   * @return {Object} A JSON object representing the serialized buffer attribute.
   */
  toJSON() {
    const e = {
      itemSize: this.itemSize,
      type: this.array.constructor.name,
      array: Array.from(this.array),
      normalized: this.normalized
    };
    return this.name !== "" && (e.name = this.name), this.usage !== Du && (e.usage = this.usage), e;
  }
}
class vd extends En {
  /**
   * Constructs a new buffer attribute.
   *
   * @param {(Array<number>|Uint16Array)} array - The array holding the attribute data.
   * @param {number} itemSize - The item size.
   * @param {boolean} [normalized=false] - Whether the data are normalized or not.
   */
  constructor(e, t, i) {
    super(new Uint16Array(e), t, i);
  }
}
class xd extends En {
  /**
   * Constructs a new buffer attribute.
   *
   * @param {(Array<number>|Uint32Array)} array - The array holding the attribute data.
   * @param {number} itemSize - The item size.
   * @param {boolean} [normalized=false] - Whether the data are normalized or not.
   */
  constructor(e, t, i) {
    super(new Uint32Array(e), t, i);
  }
}
class mt extends En {
  /**
   * Constructs a new buffer attribute.
   *
   * @param {(Array<number>|Float32Array)} array - The array holding the attribute data.
   * @param {number} itemSize - The item size.
   * @param {boolean} [normalized=false] - Whether the data are normalized or not.
   */
  constructor(e, t, i) {
    super(new Float32Array(e), t, i);
  }
}
let $g = 0;
const un = /* @__PURE__ */ new pt(), za = /* @__PURE__ */ new Tt(), ps = /* @__PURE__ */ new N(), tn = /* @__PURE__ */ new Nr(), Qs = /* @__PURE__ */ new Nr(), wt = /* @__PURE__ */ new N();
class Nt extends Ji {
  /**
   * Constructs a new geometry.
   */
  constructor() {
    super(), this.isBufferGeometry = !0, Object.defineProperty(this, "id", { value: $g++ }), this.uuid = Ur(), this.name = "", this.type = "BufferGeometry", this.index = null, this.indirect = null, this.attributes = {}, this.morphAttributes = {}, this.morphTargetsRelative = !1, this.groups = [], this.boundingBox = null, this.boundingSphere = null, this.drawRange = { start: 0, count: 1 / 0 }, this.userData = {};
  }
  /**
   * Returns the index of this geometry.
   *
   * @return {?BufferAttribute} The index. Returns `null` if no index is defined.
   */
  getIndex() {
    return this.index;
  }
  /**
   * Sets the given index to this geometry.
   *
   * @param {Array<number>|BufferAttribute} index - The index to set.
   * @return {BufferGeometry} A reference to this instance.
   */
  setIndex(e) {
    return Array.isArray(e) ? this.index = new (pd(e) ? xd : vd)(e, 1) : this.index = e, this;
  }
  /**
   * Sets the given indirect attribute to this geometry.
   *
   * @param {BufferAttribute} indirect - The attribute holding indirect draw calls.
   * @return {BufferGeometry} A reference to this instance.
   */
  setIndirect(e) {
    return this.indirect = e, this;
  }
  /**
   * Returns the indirect attribute of this geometry.
   *
   * @return {?BufferAttribute} The indirect attribute. Returns `null` if no indirect attribute is defined.
   */
  getIndirect() {
    return this.indirect;
  }
  /**
   * Returns the buffer attribute for the given name.
   *
   * @param {string} name - The attribute name.
   * @return {BufferAttribute|InterleavedBufferAttribute|undefined} The buffer attribute.
   * Returns `undefined` if not attribute has been found.
   */
  getAttribute(e) {
    return this.attributes[e];
  }
  /**
   * Sets the given attribute for the given name.
   *
   * @param {string} name - The attribute name.
   * @param {BufferAttribute|InterleavedBufferAttribute} attribute - The attribute to set.
   * @return {BufferGeometry} A reference to this instance.
   */
  setAttribute(e, t) {
    return this.attributes[e] = t, this;
  }
  /**
   * Deletes the attribute for the given name.
   *
   * @param {string} name - The attribute name to delete.
   * @return {BufferGeometry} A reference to this instance.
   */
  deleteAttribute(e) {
    return delete this.attributes[e], this;
  }
  /**
   * Returns `true` if this geometry has an attribute for the given name.
   *
   * @param {string} name - The attribute name.
   * @return {boolean} Whether this geometry has an attribute for the given name or not.
   */
  hasAttribute(e) {
    return this.attributes[e] !== void 0;
  }
  /**
   * Adds a group to this geometry.
   *
   * @param {number} start - The first element in this draw call. That is the first
   * vertex for non-indexed geometry, otherwise the first triangle index.
   * @param {number} count - Specifies how many vertices (or indices) are part of this group.
   * @param {number} [materialIndex=0] - The material array index to use.
   */
  addGroup(e, t, i = 0) {
    this.groups.push({
      start: e,
      count: t,
      materialIndex: i
    });
  }
  /**
   * Clears all groups.
   */
  clearGroups() {
    this.groups = [];
  }
  /**
   * Sets the draw range for this geometry.
   *
   * @param {number} start - The first vertex for non-indexed geometry, otherwise the first triangle index.
   * @param {number} count - For non-indexed BufferGeometry, `count` is the number of vertices to render.
   * For indexed BufferGeometry, `count` is the number of indices to render.
   */
  setDrawRange(e, t) {
    this.drawRange.start = e, this.drawRange.count = t;
  }
  /**
   * Applies the given 4x4 transformation matrix to the geometry.
   *
   * @param {Matrix4} matrix - The matrix to apply.
   * @return {BufferGeometry} A reference to this instance.
   */
  applyMatrix4(e) {
    const t = this.attributes.position;
    t !== void 0 && (t.applyMatrix4(e), t.needsUpdate = !0);
    const i = this.attributes.normal;
    if (i !== void 0) {
      const r = new qe().getNormalMatrix(e);
      i.applyNormalMatrix(r), i.needsUpdate = !0;
    }
    const s = this.attributes.tangent;
    return s !== void 0 && (s.transformDirection(e), s.needsUpdate = !0), this.boundingBox !== null && this.computeBoundingBox(), this.boundingSphere !== null && this.computeBoundingSphere(), this;
  }
  /**
   * Applies the rotation represented by the Quaternion to the geometry.
   *
   * @param {Quaternion} q - The Quaternion to apply.
   * @return {BufferGeometry} A reference to this instance.
   */
  applyQuaternion(e) {
    return un.makeRotationFromQuaternion(e), this.applyMatrix4(un), this;
  }
  /**
   * Rotates the geometry about the X axis. This is typically done as a one time
   * operation, and not during a loop. Use {@link Object3D#rotation} for typical
   * real-time mesh rotation.
   *
   * @param {number} angle - The angle in radians.
   * @return {BufferGeometry} A reference to this instance.
   */
  rotateX(e) {
    return un.makeRotationX(e), this.applyMatrix4(un), this;
  }
  /**
   * Rotates the geometry about the Y axis. This is typically done as a one time
   * operation, and not during a loop. Use {@link Object3D#rotation} for typical
   * real-time mesh rotation.
   *
   * @param {number} angle - The angle in radians.
   * @return {BufferGeometry} A reference to this instance.
   */
  rotateY(e) {
    return un.makeRotationY(e), this.applyMatrix4(un), this;
  }
  /**
   * Rotates the geometry about the Z axis. This is typically done as a one time
   * operation, and not during a loop. Use {@link Object3D#rotation} for typical
   * real-time mesh rotation.
   *
   * @param {number} angle - The angle in radians.
   * @return {BufferGeometry} A reference to this instance.
   */
  rotateZ(e) {
    return un.makeRotationZ(e), this.applyMatrix4(un), this;
  }
  /**
   * Translates the geometry. This is typically done as a one time
   * operation, and not during a loop. Use {@link Object3D#position} for typical
   * real-time mesh rotation.
   *
   * @param {number} x - The x offset.
   * @param {number} y - The y offset.
   * @param {number} z - The z offset.
   * @return {BufferGeometry} A reference to this instance.
   */
  translate(e, t, i) {
    return un.makeTranslation(e, t, i), this.applyMatrix4(un), this;
  }
  /**
   * Scales the geometry. This is typically done as a one time
   * operation, and not during a loop. Use {@link Object3D#scale} for typical
   * real-time mesh rotation.
   *
   * @param {number} x - The x scale.
   * @param {number} y - The y scale.
   * @param {number} z - The z scale.
   * @return {BufferGeometry} A reference to this instance.
   */
  scale(e, t, i) {
    return un.makeScale(e, t, i), this.applyMatrix4(un), this;
  }
  /**
   * Rotates the geometry to face a point in 3D space. This is typically done as a one time
   * operation, and not during a loop. Use {@link Object3D#lookAt} for typical
   * real-time mesh rotation.
   *
   * @param {Vector3} vector - The target point.
   * @return {BufferGeometry} A reference to this instance.
   */
  lookAt(e) {
    return za.lookAt(e), za.updateMatrix(), this.applyMatrix4(za.matrix), this;
  }
  /**
   * Center the geometry based on its bounding box.
   *
   * @return {BufferGeometry} A reference to this instance.
   */
  center() {
    return this.computeBoundingBox(), this.boundingBox.getCenter(ps).negate(), this.translate(ps.x, ps.y, ps.z), this;
  }
  /**
   * Defines a geometry by creating a `position` attribute based on the given array of points. The array
   * can hold 2D or 3D vectors. When using two-dimensional data, the `z` coordinate for all vertices is
   * set to `0`.
   *
   * If the method is used with an existing `position` attribute, the vertex data are overwritten with the
   * data from the array. The length of the array must match the vertex count.
   *
   * @param {Array<Vector2>|Array<Vector3>} points - The points.
   * @return {BufferGeometry} A reference to this instance.
   */
  setFromPoints(e) {
    const t = this.getAttribute("position");
    if (t === void 0) {
      const i = [];
      for (let s = 0, r = e.length; s < r; s++) {
        const o = e[s];
        i.push(o.x, o.y, o.z || 0);
      }
      this.setAttribute("position", new mt(i, 3));
    } else {
      const i = Math.min(e.length, t.count);
      for (let s = 0; s < i; s++) {
        const r = e[s];
        t.setXYZ(s, r.x, r.y, r.z || 0);
      }
      e.length > t.count && console.warn("THREE.BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."), t.needsUpdate = !0;
    }
    return this;
  }
  /**
   * Computes the bounding box of the geometry, and updates the `boundingBox` member.
   * The bounding box is not computed by the engine; it must be computed by your app.
   * You may need to recompute the bounding box if the geometry vertices are modified.
   */
  computeBoundingBox() {
    this.boundingBox === null && (this.boundingBox = new Nr());
    const e = this.attributes.position, t = this.morphAttributes.position;
    if (e && e.isGLBufferAttribute) {
      console.error("THREE.BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.", this), this.boundingBox.set(
        new N(-1 / 0, -1 / 0, -1 / 0),
        new N(1 / 0, 1 / 0, 1 / 0)
      );
      return;
    }
    if (e !== void 0) {
      if (this.boundingBox.setFromBufferAttribute(e), t)
        for (let i = 0, s = t.length; i < s; i++) {
          const r = t[i];
          tn.setFromBufferAttribute(r), this.morphTargetsRelative ? (wt.addVectors(this.boundingBox.min, tn.min), this.boundingBox.expandByPoint(wt), wt.addVectors(this.boundingBox.max, tn.max), this.boundingBox.expandByPoint(wt)) : (this.boundingBox.expandByPoint(tn.min), this.boundingBox.expandByPoint(tn.max));
        }
    } else
      this.boundingBox.makeEmpty();
    (isNaN(this.boundingBox.min.x) || isNaN(this.boundingBox.min.y) || isNaN(this.boundingBox.min.z)) && console.error('THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.', this);
  }
  /**
   * Computes the bounding sphere of the geometry, and updates the `boundingSphere` member.
   * The engine automatically computes the bounding sphere when it is needed, e.g., for ray casting or view frustum culling.
   * You may need to recompute the bounding sphere if the geometry vertices are modified.
   */
  computeBoundingSphere() {
    this.boundingSphere === null && (this.boundingSphere = new Fr());
    const e = this.attributes.position, t = this.morphAttributes.position;
    if (e && e.isGLBufferAttribute) {
      console.error("THREE.BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.", this), this.boundingSphere.set(new N(), 1 / 0);
      return;
    }
    if (e) {
      const i = this.boundingSphere.center;
      if (tn.setFromBufferAttribute(e), t)
        for (let r = 0, o = t.length; r < o; r++) {
          const a = t[r];
          Qs.setFromBufferAttribute(a), this.morphTargetsRelative ? (wt.addVectors(tn.min, Qs.min), tn.expandByPoint(wt), wt.addVectors(tn.max, Qs.max), tn.expandByPoint(wt)) : (tn.expandByPoint(Qs.min), tn.expandByPoint(Qs.max));
        }
      tn.getCenter(i);
      let s = 0;
      for (let r = 0, o = e.count; r < o; r++)
        wt.fromBufferAttribute(e, r), s = Math.max(s, i.distanceToSquared(wt));
      if (t)
        for (let r = 0, o = t.length; r < o; r++) {
          const a = t[r], l = this.morphTargetsRelative;
          for (let c = 0, u = a.count; c < u; c++)
            wt.fromBufferAttribute(a, c), l && (ps.fromBufferAttribute(e, c), wt.add(ps)), s = Math.max(s, i.distanceToSquared(wt));
        }
      this.boundingSphere.radius = Math.sqrt(s), isNaN(this.boundingSphere.radius) && console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.', this);
    }
  }
  /**
   * Calculates and adds a tangent attribute to this geometry.
   *
   * The computation is only supported for indexed geometries and if position, normal, and uv attributes
   * are defined. When using a tangent space normal map, prefer the MikkTSpace algorithm provided by
   * {@link BufferGeometryUtils#computeMikkTSpaceTangents} instead.
   */
  computeTangents() {
    const e = this.index, t = this.attributes;
    if (e === null || t.position === void 0 || t.normal === void 0 || t.uv === void 0) {
      console.error("THREE.BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");
      return;
    }
    const i = t.position, s = t.normal, r = t.uv;
    this.hasAttribute("tangent") === !1 && this.setAttribute("tangent", new En(new Float32Array(4 * i.count), 4));
    const o = this.getAttribute("tangent"), a = [], l = [];
    for (let U = 0; U < i.count; U++)
      a[U] = new N(), l[U] = new N();
    const c = new N(), u = new N(), h = new N(), f = new Ve(), p = new Ve(), v = new Ve(), x = new N(), m = new N();
    function d(U, S, y) {
      c.fromBufferAttribute(i, U), u.fromBufferAttribute(i, S), h.fromBufferAttribute(i, y), f.fromBufferAttribute(r, U), p.fromBufferAttribute(r, S), v.fromBufferAttribute(r, y), u.sub(c), h.sub(c), p.sub(f), v.sub(f);
      const D = 1 / (p.x * v.y - v.x * p.y);
      isFinite(D) && (x.copy(u).multiplyScalar(v.y).addScaledVector(h, -p.y).multiplyScalar(D), m.copy(h).multiplyScalar(p.x).addScaledVector(u, -v.x).multiplyScalar(D), a[U].add(x), a[S].add(x), a[y].add(x), l[U].add(m), l[S].add(m), l[y].add(m));
    }
    let b = this.groups;
    b.length === 0 && (b = [{
      start: 0,
      count: e.count
    }]);
    for (let U = 0, S = b.length; U < S; ++U) {
      const y = b[U], D = y.start, L = y.count;
      for (let V = D, Z = D + L; V < Z; V += 3)
        d(
          e.getX(V + 0),
          e.getX(V + 1),
          e.getX(V + 2)
        );
    }
    const A = new N(), M = new N(), C = new N(), w = new N();
    function P(U) {
      C.fromBufferAttribute(s, U), w.copy(C);
      const S = a[U];
      A.copy(S), A.sub(C.multiplyScalar(C.dot(S))).normalize(), M.crossVectors(w, S);
      const D = M.dot(l[U]) < 0 ? -1 : 1;
      o.setXYZW(U, A.x, A.y, A.z, D);
    }
    for (let U = 0, S = b.length; U < S; ++U) {
      const y = b[U], D = y.start, L = y.count;
      for (let V = D, Z = D + L; V < Z; V += 3)
        P(e.getX(V + 0)), P(e.getX(V + 1)), P(e.getX(V + 2));
    }
  }
  /**
   * Computes vertex normals for the given vertex data. For indexed geometries, the method sets
   * each vertex normal to be the average of the face normals of the faces that share that vertex.
   * For non-indexed geometries, vertices are not shared, and the method sets each vertex normal
   * to be the same as the face normal.
   */
  computeVertexNormals() {
    const e = this.index, t = this.getAttribute("position");
    if (t !== void 0) {
      let i = this.getAttribute("normal");
      if (i === void 0)
        i = new En(new Float32Array(t.count * 3), 3), this.setAttribute("normal", i);
      else
        for (let f = 0, p = i.count; f < p; f++)
          i.setXYZ(f, 0, 0, 0);
      const s = new N(), r = new N(), o = new N(), a = new N(), l = new N(), c = new N(), u = new N(), h = new N();
      if (e)
        for (let f = 0, p = e.count; f < p; f += 3) {
          const v = e.getX(f + 0), x = e.getX(f + 1), m = e.getX(f + 2);
          s.fromBufferAttribute(t, v), r.fromBufferAttribute(t, x), o.fromBufferAttribute(t, m), u.subVectors(o, r), h.subVectors(s, r), u.cross(h), a.fromBufferAttribute(i, v), l.fromBufferAttribute(i, x), c.fromBufferAttribute(i, m), a.add(u), l.add(u), c.add(u), i.setXYZ(v, a.x, a.y, a.z), i.setXYZ(x, l.x, l.y, l.z), i.setXYZ(m, c.x, c.y, c.z);
        }
      else
        for (let f = 0, p = t.count; f < p; f += 3)
          s.fromBufferAttribute(t, f + 0), r.fromBufferAttribute(t, f + 1), o.fromBufferAttribute(t, f + 2), u.subVectors(o, r), h.subVectors(s, r), u.cross(h), i.setXYZ(f + 0, u.x, u.y, u.z), i.setXYZ(f + 1, u.x, u.y, u.z), i.setXYZ(f + 2, u.x, u.y, u.z);
      this.normalizeNormals(), i.needsUpdate = !0;
    }
  }
  /**
   * Ensures every normal vector in a geometry will have a magnitude of `1`. This will
   * correct lighting on the geometry surfaces.
   */
  normalizeNormals() {
    const e = this.attributes.normal;
    for (let t = 0, i = e.count; t < i; t++)
      wt.fromBufferAttribute(e, t), wt.normalize(), e.setXYZ(t, wt.x, wt.y, wt.z);
  }
  /**
   * Return a new non-index version of this indexed geometry. If the geometry
   * is already non-indexed, the method is a NOOP.
   *
   * @return {BufferGeometry} The non-indexed version of this indexed geometry.
   */
  toNonIndexed() {
    function e(a, l) {
      const c = a.array, u = a.itemSize, h = a.normalized, f = new c.constructor(l.length * u);
      let p = 0, v = 0;
      for (let x = 0, m = l.length; x < m; x++) {
        a.isInterleavedBufferAttribute ? p = l[x] * a.data.stride + a.offset : p = l[x] * u;
        for (let d = 0; d < u; d++)
          f[v++] = c[p++];
      }
      return new En(f, u, h);
    }
    if (this.index === null)
      return console.warn("THREE.BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."), this;
    const t = new Nt(), i = this.index.array, s = this.attributes;
    for (const a in s) {
      const l = s[a], c = e(l, i);
      t.setAttribute(a, c);
    }
    const r = this.morphAttributes;
    for (const a in r) {
      const l = [], c = r[a];
      for (let u = 0, h = c.length; u < h; u++) {
        const f = c[u], p = e(f, i);
        l.push(p);
      }
      t.morphAttributes[a] = l;
    }
    t.morphTargetsRelative = this.morphTargetsRelative;
    const o = this.groups;
    for (let a = 0, l = o.length; a < l; a++) {
      const c = o[a];
      t.addGroup(c.start, c.count, c.materialIndex);
    }
    return t;
  }
  /**
   * Serializes the geometry into JSON.
   *
   * @return {Object} A JSON object representing the serialized geometry.
   */
  toJSON() {
    const e = {
      metadata: {
        version: 4.7,
        type: "BufferGeometry",
        generator: "BufferGeometry.toJSON"
      }
    };
    if (e.uuid = this.uuid, e.type = this.type, this.name !== "" && (e.name = this.name), Object.keys(this.userData).length > 0 && (e.userData = this.userData), this.parameters !== void 0) {
      const l = this.parameters;
      for (const c in l)
        l[c] !== void 0 && (e[c] = l[c]);
      return e;
    }
    e.data = { attributes: {} };
    const t = this.index;
    t !== null && (e.data.index = {
      type: t.array.constructor.name,
      array: Array.prototype.slice.call(t.array)
    });
    const i = this.attributes;
    for (const l in i) {
      const c = i[l];
      e.data.attributes[l] = c.toJSON(e.data);
    }
    const s = {};
    let r = !1;
    for (const l in this.morphAttributes) {
      const c = this.morphAttributes[l], u = [];
      for (let h = 0, f = c.length; h < f; h++) {
        const p = c[h];
        u.push(p.toJSON(e.data));
      }
      u.length > 0 && (s[l] = u, r = !0);
    }
    r && (e.data.morphAttributes = s, e.data.morphTargetsRelative = this.morphTargetsRelative);
    const o = this.groups;
    o.length > 0 && (e.data.groups = JSON.parse(JSON.stringify(o)));
    const a = this.boundingSphere;
    return a !== null && (e.data.boundingSphere = a.toJSON()), e;
  }
  /**
   * Returns a new geometry with copied values from this instance.
   *
   * @return {BufferGeometry} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Copies the values of the given geometry to this instance.
   *
   * @param {BufferGeometry} source - The geometry to copy.
   * @return {BufferGeometry} A reference to this instance.
   */
  copy(e) {
    this.index = null, this.attributes = {}, this.morphAttributes = {}, this.groups = [], this.boundingBox = null, this.boundingSphere = null;
    const t = {};
    this.name = e.name;
    const i = e.index;
    i !== null && this.setIndex(i.clone());
    const s = e.attributes;
    for (const c in s) {
      const u = s[c];
      this.setAttribute(c, u.clone(t));
    }
    const r = e.morphAttributes;
    for (const c in r) {
      const u = [], h = r[c];
      for (let f = 0, p = h.length; f < p; f++)
        u.push(h[f].clone(t));
      this.morphAttributes[c] = u;
    }
    this.morphTargetsRelative = e.morphTargetsRelative;
    const o = e.groups;
    for (let c = 0, u = o.length; c < u; c++) {
      const h = o[c];
      this.addGroup(h.start, h.count, h.materialIndex);
    }
    const a = e.boundingBox;
    a !== null && (this.boundingBox = a.clone());
    const l = e.boundingSphere;
    return l !== null && (this.boundingSphere = l.clone()), this.drawRange.start = e.drawRange.start, this.drawRange.count = e.drawRange.count, this.userData = e.userData, this;
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   *
   * @fires BufferGeometry#dispose
   */
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
}
const Xu = /* @__PURE__ */ new pt(), Li = /* @__PURE__ */ new na(), Jr = /* @__PURE__ */ new Fr(), Yu = /* @__PURE__ */ new N(), Qr = /* @__PURE__ */ new N(), eo = /* @__PURE__ */ new N(), to = /* @__PURE__ */ new N(), Ha = /* @__PURE__ */ new N(), no = /* @__PURE__ */ new N(), qu = /* @__PURE__ */ new N(), io = /* @__PURE__ */ new N();
class vt extends Tt {
  /**
   * Constructs a new mesh.
   *
   * @param {BufferGeometry} [geometry] - The mesh geometry.
   * @param {Material|Array<Material>} [material] - The mesh material.
   */
  constructor(e = new Nt(), t = new Rn()) {
    super(), this.isMesh = !0, this.type = "Mesh", this.geometry = e, this.material = t, this.morphTargetDictionary = void 0, this.morphTargetInfluences = void 0, this.count = 1, this.updateMorphTargets();
  }
  copy(e, t) {
    return super.copy(e, t), e.morphTargetInfluences !== void 0 && (this.morphTargetInfluences = e.morphTargetInfluences.slice()), e.morphTargetDictionary !== void 0 && (this.morphTargetDictionary = Object.assign({}, e.morphTargetDictionary)), this.material = Array.isArray(e.material) ? e.material.slice() : e.material, this.geometry = e.geometry, this;
  }
  /**
   * Sets the values of {@link Mesh#morphTargetDictionary} and {@link Mesh#morphTargetInfluences}
   * to make sure existing morph targets can influence this 3D object.
   */
  updateMorphTargets() {
    const t = this.geometry.morphAttributes, i = Object.keys(t);
    if (i.length > 0) {
      const s = t[i[0]];
      if (s !== void 0) {
        this.morphTargetInfluences = [], this.morphTargetDictionary = {};
        for (let r = 0, o = s.length; r < o; r++) {
          const a = s[r].name || String(r);
          this.morphTargetInfluences.push(0), this.morphTargetDictionary[a] = r;
        }
      }
    }
  }
  /**
   * Returns the local-space position of the vertex at the given index, taking into
   * account the current animation state of both morph targets and skinning.
   *
   * @param {number} index - The vertex index.
   * @param {Vector3} target - The target object that is used to store the method's result.
   * @return {Vector3} The vertex position in local space.
   */
  getVertexPosition(e, t) {
    const i = this.geometry, s = i.attributes.position, r = i.morphAttributes.position, o = i.morphTargetsRelative;
    t.fromBufferAttribute(s, e);
    const a = this.morphTargetInfluences;
    if (r && a) {
      no.set(0, 0, 0);
      for (let l = 0, c = r.length; l < c; l++) {
        const u = a[l], h = r[l];
        u !== 0 && (Ha.fromBufferAttribute(h, e), o ? no.addScaledVector(Ha, u) : no.addScaledVector(Ha.sub(t), u));
      }
      t.add(no);
    }
    return t;
  }
  /**
   * Computes intersection points between a casted ray and this line.
   *
   * @param {Raycaster} raycaster - The raycaster.
   * @param {Array<Object>} intersects - The target array that holds the intersection points.
   */
  raycast(e, t) {
    const i = this.geometry, s = this.material, r = this.matrixWorld;
    s !== void 0 && (i.boundingSphere === null && i.computeBoundingSphere(), Jr.copy(i.boundingSphere), Jr.applyMatrix4(r), Li.copy(e.ray).recast(e.near), !(Jr.containsPoint(Li.origin) === !1 && (Li.intersectSphere(Jr, Yu) === null || Li.origin.distanceToSquared(Yu) > (e.far - e.near) ** 2)) && (Xu.copy(r).invert(), Li.copy(e.ray).applyMatrix4(Xu), !(i.boundingBox !== null && Li.intersectsBox(i.boundingBox) === !1) && this._computeIntersections(e, t, Li)));
  }
  _computeIntersections(e, t, i) {
    let s;
    const r = this.geometry, o = this.material, a = r.index, l = r.attributes.position, c = r.attributes.uv, u = r.attributes.uv1, h = r.attributes.normal, f = r.groups, p = r.drawRange;
    if (a !== null)
      if (Array.isArray(o))
        for (let v = 0, x = f.length; v < x; v++) {
          const m = f[v], d = o[m.materialIndex], b = Math.max(m.start, p.start), A = Math.min(a.count, Math.min(m.start + m.count, p.start + p.count));
          for (let M = b, C = A; M < C; M += 3) {
            const w = a.getX(M), P = a.getX(M + 1), U = a.getX(M + 2);
            s = so(this, d, e, i, c, u, h, w, P, U), s && (s.faceIndex = Math.floor(M / 3), s.face.materialIndex = m.materialIndex, t.push(s));
          }
        }
      else {
        const v = Math.max(0, p.start), x = Math.min(a.count, p.start + p.count);
        for (let m = v, d = x; m < d; m += 3) {
          const b = a.getX(m), A = a.getX(m + 1), M = a.getX(m + 2);
          s = so(this, o, e, i, c, u, h, b, A, M), s && (s.faceIndex = Math.floor(m / 3), t.push(s));
        }
      }
    else if (l !== void 0)
      if (Array.isArray(o))
        for (let v = 0, x = f.length; v < x; v++) {
          const m = f[v], d = o[m.materialIndex], b = Math.max(m.start, p.start), A = Math.min(l.count, Math.min(m.start + m.count, p.start + p.count));
          for (let M = b, C = A; M < C; M += 3) {
            const w = M, P = M + 1, U = M + 2;
            s = so(this, d, e, i, c, u, h, w, P, U), s && (s.faceIndex = Math.floor(M / 3), s.face.materialIndex = m.materialIndex, t.push(s));
          }
        }
      else {
        const v = Math.max(0, p.start), x = Math.min(l.count, p.start + p.count);
        for (let m = v, d = x; m < d; m += 3) {
          const b = m, A = m + 1, M = m + 2;
          s = so(this, o, e, i, c, u, h, b, A, M), s && (s.faceIndex = Math.floor(m / 3), t.push(s));
        }
      }
  }
}
function Zg(n, e, t, i, s, r, o, a) {
  let l;
  if (e.side === Wt ? l = i.intersectTriangle(o, r, s, !0, a) : l = i.intersectTriangle(s, r, o, e.side === yi, a), l === null) return null;
  io.copy(a), io.applyMatrix4(n.matrixWorld);
  const c = t.ray.origin.distanceTo(io);
  return c < t.near || c > t.far ? null : {
    distance: c,
    point: io.clone(),
    object: n
  };
}
function so(n, e, t, i, s, r, o, a, l, c) {
  n.getVertexPosition(a, Qr), n.getVertexPosition(l, eo), n.getVertexPosition(c, to);
  const u = Zg(n, e, t, i, Qr, eo, to, qu);
  if (u) {
    const h = new N();
    fn.getBarycoord(qu, Qr, eo, to, h), s && (u.uv = fn.getInterpolatedAttribute(s, a, l, c, h, new Ve())), r && (u.uv1 = fn.getInterpolatedAttribute(r, a, l, c, h, new Ve())), o && (u.normal = fn.getInterpolatedAttribute(o, a, l, c, h, new N()), u.normal.dot(i.direction) > 0 && u.normal.multiplyScalar(-1));
    const f = {
      a,
      b: l,
      c,
      normal: new N(),
      materialIndex: 0
    };
    fn.getNormal(Qr, eo, to, f.normal), u.face = f, u.barycoord = h;
  }
  return u;
}
class Ki extends Nt {
  /**
   * Constructs a new box geometry.
   *
   * @param {number} [width=1] - The width. That is, the length of the edges parallel to the X axis.
   * @param {number} [height=1] - The height. That is, the length of the edges parallel to the Y axis.
   * @param {number} [depth=1] - The depth. That is, the length of the edges parallel to the Z axis.
   * @param {number} [widthSegments=1] - Number of segmented rectangular faces along the width of the sides.
   * @param {number} [heightSegments=1] - Number of segmented rectangular faces along the height of the sides.
   * @param {number} [depthSegments=1] - Number of segmented rectangular faces along the depth of the sides.
   */
  constructor(e = 1, t = 1, i = 1, s = 1, r = 1, o = 1) {
    super(), this.type = "BoxGeometry", this.parameters = {
      width: e,
      height: t,
      depth: i,
      widthSegments: s,
      heightSegments: r,
      depthSegments: o
    };
    const a = this;
    s = Math.floor(s), r = Math.floor(r), o = Math.floor(o);
    const l = [], c = [], u = [], h = [];
    let f = 0, p = 0;
    v("z", "y", "x", -1, -1, i, t, e, o, r, 0), v("z", "y", "x", 1, -1, i, t, -e, o, r, 1), v("x", "z", "y", 1, 1, e, i, t, s, o, 2), v("x", "z", "y", 1, -1, e, i, -t, s, o, 3), v("x", "y", "z", 1, -1, e, t, i, s, r, 4), v("x", "y", "z", -1, -1, e, t, -i, s, r, 5), this.setIndex(l), this.setAttribute("position", new mt(c, 3)), this.setAttribute("normal", new mt(u, 3)), this.setAttribute("uv", new mt(h, 2));
    function v(x, m, d, b, A, M, C, w, P, U, S) {
      const y = M / P, D = C / U, L = M / 2, V = C / 2, Z = w / 2, ne = P + 1, J = U + 1;
      let ie = 0, H = 0;
      const fe = new N();
      for (let ge = 0; ge < J; ge++) {
        const ye = ge * D - V;
        for (let Fe = 0; Fe < ne; Fe++) {
          const Je = Fe * y - L;
          fe[x] = Je * b, fe[m] = ye * A, fe[d] = Z, c.push(fe.x, fe.y, fe.z), fe[x] = 0, fe[m] = 0, fe[d] = w > 0 ? 1 : -1, u.push(fe.x, fe.y, fe.z), h.push(Fe / P), h.push(1 - ge / U), ie += 1;
        }
      }
      for (let ge = 0; ge < U; ge++)
        for (let ye = 0; ye < P; ye++) {
          const Fe = f + ye + ne * ge, Je = f + ye + ne * (ge + 1), Ge = f + (ye + 1) + ne * (ge + 1), Ae = f + (ye + 1) + ne * ge;
          l.push(Fe, Je, Ae), l.push(Je, Ge, Ae), H += 6;
        }
      a.addGroup(p, H, S), p += H, f += ie;
    }
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {BoxGeometry} A new instance.
   */
  static fromJSON(e) {
    return new Ki(e.width, e.height, e.depth, e.widthSegments, e.heightSegments, e.depthSegments);
  }
}
function zs(n) {
  const e = {};
  for (const t in n) {
    e[t] = {};
    for (const i in n[t]) {
      const s = n[t][i];
      s && (s.isColor || s.isMatrix3 || s.isMatrix4 || s.isVector2 || s.isVector3 || s.isVector4 || s.isTexture || s.isQuaternion) ? s.isRenderTargetTexture ? (console.warn("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."), e[t][i] = null) : e[t][i] = s.clone() : Array.isArray(s) ? e[t][i] = s.slice() : e[t][i] = s;
    }
  }
  return e;
}
function zt(n) {
  const e = {};
  for (let t = 0; t < n.length; t++) {
    const i = zs(n[t]);
    for (const s in i)
      e[s] = i[s];
  }
  return e;
}
function Jg(n) {
  const e = [];
  for (let t = 0; t < n.length; t++)
    e.push(n[t].clone());
  return e;
}
function Md(n) {
  const e = n.getRenderTarget();
  return e === null ? n.outputColorSpace : e.isXRRenderTarget === !0 ? e.texture.colorSpace : et.workingColorSpace;
}
const Qg = { clone: zs, merge: zt };
var e0 = `void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`, t0 = `void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;
class Ei extends Qi {
  /**
   * Constructs a new shader material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isShaderMaterial = !0, this.type = "ShaderMaterial", this.defines = {}, this.uniforms = {}, this.uniformsGroups = [], this.vertexShader = e0, this.fragmentShader = t0, this.linewidth = 1, this.wireframe = !1, this.wireframeLinewidth = 1, this.fog = !1, this.lights = !1, this.clipping = !1, this.forceSinglePass = !0, this.extensions = {
      clipCullDistance: !1,
      // set to use vertex shader clipping
      multiDraw: !1
      // set to use vertex shader multi_draw / enable gl_DrawID
    }, this.defaultAttributeValues = {
      color: [1, 1, 1],
      uv: [0, 0],
      uv1: [0, 0]
    }, this.index0AttributeName = void 0, this.uniformsNeedUpdate = !1, this.glslVersion = null, e !== void 0 && this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.fragmentShader = e.fragmentShader, this.vertexShader = e.vertexShader, this.uniforms = zs(e.uniforms), this.uniformsGroups = Jg(e.uniformsGroups), this.defines = Object.assign({}, e.defines), this.wireframe = e.wireframe, this.wireframeLinewidth = e.wireframeLinewidth, this.fog = e.fog, this.lights = e.lights, this.clipping = e.clipping, this.extensions = Object.assign({}, e.extensions), this.glslVersion = e.glslVersion, this;
  }
  toJSON(e) {
    const t = super.toJSON(e);
    t.glslVersion = this.glslVersion, t.uniforms = {};
    for (const s in this.uniforms) {
      const o = this.uniforms[s].value;
      o && o.isTexture ? t.uniforms[s] = {
        type: "t",
        value: o.toJSON(e).uuid
      } : o && o.isColor ? t.uniforms[s] = {
        type: "c",
        value: o.getHex()
      } : o && o.isVector2 ? t.uniforms[s] = {
        type: "v2",
        value: o.toArray()
      } : o && o.isVector3 ? t.uniforms[s] = {
        type: "v3",
        value: o.toArray()
      } : o && o.isVector4 ? t.uniforms[s] = {
        type: "v4",
        value: o.toArray()
      } : o && o.isMatrix3 ? t.uniforms[s] = {
        type: "m3",
        value: o.toArray()
      } : o && o.isMatrix4 ? t.uniforms[s] = {
        type: "m4",
        value: o.toArray()
      } : t.uniforms[s] = {
        value: o
      };
    }
    Object.keys(this.defines).length > 0 && (t.defines = this.defines), t.vertexShader = this.vertexShader, t.fragmentShader = this.fragmentShader, t.lights = this.lights, t.clipping = this.clipping;
    const i = {};
    for (const s in this.extensions)
      this.extensions[s] === !0 && (i[s] = !0);
    return Object.keys(i).length > 0 && (t.extensions = i), t;
  }
}
class Sd extends Tt {
  /**
   * Constructs a new camera.
   */
  constructor() {
    super(), this.isCamera = !0, this.type = "Camera", this.matrixWorldInverse = new pt(), this.projectionMatrix = new pt(), this.projectionMatrixInverse = new pt(), this.coordinateSystem = Nn, this._reversedDepth = !1;
  }
  /**
   * The flag that indicates whether the camera uses a reversed depth buffer.
   *
   * @type {boolean}
   * @default false
   */
  get reversedDepth() {
    return this._reversedDepth;
  }
  copy(e, t) {
    return super.copy(e, t), this.matrixWorldInverse.copy(e.matrixWorldInverse), this.projectionMatrix.copy(e.projectionMatrix), this.projectionMatrixInverse.copy(e.projectionMatrixInverse), this.coordinateSystem = e.coordinateSystem, this;
  }
  /**
   * Returns a vector representing the ("look") direction of the 3D object in world space.
   *
   * This method is overwritten since cameras have a different forward vector compared to other
   * 3D objects. A camera looks down its local, negative z-axis by default.
   *
   * @param {Vector3} target - The target vector the result is stored to.
   * @return {Vector3} The 3D object's direction in world space.
   */
  getWorldDirection(e) {
    return super.getWorldDirection(e).negate();
  }
  updateMatrixWorld(e) {
    super.updateMatrixWorld(e), this.matrixWorldInverse.copy(this.matrixWorld).invert();
  }
  updateWorldMatrix(e, t) {
    super.updateWorldMatrix(e, t), this.matrixWorldInverse.copy(this.matrixWorld).invert();
  }
  clone() {
    return new this.constructor().copy(this);
  }
}
const fi = /* @__PURE__ */ new N(), ju = /* @__PURE__ */ new Ve(), Ku = /* @__PURE__ */ new Ve();
class rn extends Sd {
  /**
   * Constructs a new perspective camera.
   *
   * @param {number} [fov=50] - The vertical field of view.
   * @param {number} [aspect=1] - The aspect ratio.
   * @param {number} [near=0.1] - The camera's near plane.
   * @param {number} [far=2000] - The camera's far plane.
   */
  constructor(e = 50, t = 1, i = 0.1, s = 2e3) {
    super(), this.isPerspectiveCamera = !0, this.type = "PerspectiveCamera", this.fov = e, this.zoom = 1, this.near = i, this.far = s, this.focus = 10, this.aspect = t, this.view = null, this.filmGauge = 35, this.filmOffset = 0, this.updateProjectionMatrix();
  }
  copy(e, t) {
    return super.copy(e, t), this.fov = e.fov, this.zoom = e.zoom, this.near = e.near, this.far = e.far, this.focus = e.focus, this.aspect = e.aspect, this.view = e.view === null ? null : Object.assign({}, e.view), this.filmGauge = e.filmGauge, this.filmOffset = e.filmOffset, this;
  }
  /**
   * Sets the FOV by focal length in respect to the current {@link PerspectiveCamera#filmGauge}.
   *
   * The default film gauge is 35, so that the focal length can be specified for
   * a 35mm (full frame) camera.
   *
   * @param {number} focalLength - Values for focal length and film gauge must have the same unit.
   */
  setFocalLength(e) {
    const t = 0.5 * this.getFilmHeight() / e;
    this.fov = Ql * 2 * Math.atan(t), this.updateProjectionMatrix();
  }
  /**
   * Returns the focal length from the current {@link PerspectiveCamera#fov} and
   * {@link PerspectiveCamera#filmGauge}.
   *
   * @return {number} The computed focal length.
   */
  getFocalLength() {
    const e = Math.tan(pr * 0.5 * this.fov);
    return 0.5 * this.getFilmHeight() / e;
  }
  /**
   * Returns the current vertical field of view angle in degrees considering {@link PerspectiveCamera#zoom}.
   *
   * @return {number} The effective FOV.
   */
  getEffectiveFOV() {
    return Ql * 2 * Math.atan(
      Math.tan(pr * 0.5 * this.fov) / this.zoom
    );
  }
  /**
   * Returns the width of the image on the film. If {@link PerspectiveCamera#aspect} is greater than or
   * equal to one (landscape format), the result equals {@link PerspectiveCamera#filmGauge}.
   *
   * @return {number} The film width.
   */
  getFilmWidth() {
    return this.filmGauge * Math.min(this.aspect, 1);
  }
  /**
   * Returns the height of the image on the film. If {@link PerspectiveCamera#aspect} is greater than or
   * equal to one (landscape format), the result equals {@link PerspectiveCamera#filmGauge}.
   *
   * @return {number} The film width.
   */
  getFilmHeight() {
    return this.filmGauge / Math.max(this.aspect, 1);
  }
  /**
   * Computes the 2D bounds of the camera's viewable rectangle at a given distance along the viewing direction.
   * Sets `minTarget` and `maxTarget` to the coordinates of the lower-left and upper-right corners of the view rectangle.
   *
   * @param {number} distance - The viewing distance.
   * @param {Vector2} minTarget - The lower-left corner of the view rectangle is written into this vector.
   * @param {Vector2} maxTarget - The upper-right corner of the view rectangle is written into this vector.
   */
  getViewBounds(e, t, i) {
    fi.set(-1, -1, 0.5).applyMatrix4(this.projectionMatrixInverse), t.set(fi.x, fi.y).multiplyScalar(-e / fi.z), fi.set(1, 1, 0.5).applyMatrix4(this.projectionMatrixInverse), i.set(fi.x, fi.y).multiplyScalar(-e / fi.z);
  }
  /**
   * Computes the width and height of the camera's viewable rectangle at a given distance along the viewing direction.
   *
   * @param {number} distance - The viewing distance.
   * @param {Vector2} target - The target vector that is used to store result where x is width and y is height.
   * @returns {Vector2} The view size.
   */
  getViewSize(e, t) {
    return this.getViewBounds(e, ju, Ku), t.subVectors(Ku, ju);
  }
  /**
   * Sets an offset in a larger frustum. This is useful for multi-window or
   * multi-monitor/multi-machine setups.
   *
   * For example, if you have 3x2 monitors and each monitor is 1920x1080 and
   * the monitors are in grid like this
   *```
   *   +---+---+---+
   *   | A | B | C |
   *   +---+---+---+
   *   | D | E | F |
   *   +---+---+---+
   *```
   * then for each monitor you would call it like this:
   *```js
   * const w = 1920;
   * const h = 1080;
   * const fullWidth = w * 3;
   * const fullHeight = h * 2;
   *
   * // --A--
   * camera.setViewOffset( fullWidth, fullHeight, w * 0, h * 0, w, h );
   * // --B--
   * camera.setViewOffset( fullWidth, fullHeight, w * 1, h * 0, w, h );
   * // --C--
   * camera.setViewOffset( fullWidth, fullHeight, w * 2, h * 0, w, h );
   * // --D--
   * camera.setViewOffset( fullWidth, fullHeight, w * 0, h * 1, w, h );
   * // --E--
   * camera.setViewOffset( fullWidth, fullHeight, w * 1, h * 1, w, h );
   * // --F--
   * camera.setViewOffset( fullWidth, fullHeight, w * 2, h * 1, w, h );
   * ```
   *
   * Note there is no reason monitors have to be the same size or in a grid.
   *
   * @param {number} fullWidth - The full width of multiview setup.
   * @param {number} fullHeight - The full height of multiview setup.
   * @param {number} x - The horizontal offset of the subcamera.
   * @param {number} y - The vertical offset of the subcamera.
   * @param {number} width - The width of subcamera.
   * @param {number} height - The height of subcamera.
   */
  setViewOffset(e, t, i, s, r, o) {
    this.aspect = e / t, this.view === null && (this.view = {
      enabled: !0,
      fullWidth: 1,
      fullHeight: 1,
      offsetX: 0,
      offsetY: 0,
      width: 1,
      height: 1
    }), this.view.enabled = !0, this.view.fullWidth = e, this.view.fullHeight = t, this.view.offsetX = i, this.view.offsetY = s, this.view.width = r, this.view.height = o, this.updateProjectionMatrix();
  }
  /**
   * Removes the view offset from the projection matrix.
   */
  clearViewOffset() {
    this.view !== null && (this.view.enabled = !1), this.updateProjectionMatrix();
  }
  /**
   * Updates the camera's projection matrix. Must be called after any change of
   * camera properties.
   */
  updateProjectionMatrix() {
    const e = this.near;
    let t = e * Math.tan(pr * 0.5 * this.fov) / this.zoom, i = 2 * t, s = this.aspect * i, r = -0.5 * s;
    const o = this.view;
    if (this.view !== null && this.view.enabled) {
      const l = o.fullWidth, c = o.fullHeight;
      r += o.offsetX * s / l, t -= o.offsetY * i / c, s *= o.width / l, i *= o.height / c;
    }
    const a = this.filmOffset;
    a !== 0 && (r += e * a / this.getFilmWidth()), this.projectionMatrix.makePerspective(r, r + s, t, t - i, e, this.far, this.coordinateSystem, this.reversedDepth), this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
  }
  toJSON(e) {
    const t = super.toJSON(e);
    return t.object.fov = this.fov, t.object.zoom = this.zoom, t.object.near = this.near, t.object.far = this.far, t.object.focus = this.focus, t.object.aspect = this.aspect, this.view !== null && (t.object.view = Object.assign({}, this.view)), t.object.filmGauge = this.filmGauge, t.object.filmOffset = this.filmOffset, t;
  }
}
const ms = -90, _s = 1;
class n0 extends Tt {
  /**
   * Constructs a new cube camera.
   *
   * @param {number} near - The camera's near plane.
   * @param {number} far - The camera's far plane.
   * @param {WebGLCubeRenderTarget} renderTarget - The cube render target.
   */
  constructor(e, t, i) {
    super(), this.type = "CubeCamera", this.renderTarget = i, this.coordinateSystem = null, this.activeMipmapLevel = 0;
    const s = new rn(ms, _s, e, t);
    s.layers = this.layers, this.add(s);
    const r = new rn(ms, _s, e, t);
    r.layers = this.layers, this.add(r);
    const o = new rn(ms, _s, e, t);
    o.layers = this.layers, this.add(o);
    const a = new rn(ms, _s, e, t);
    a.layers = this.layers, this.add(a);
    const l = new rn(ms, _s, e, t);
    l.layers = this.layers, this.add(l);
    const c = new rn(ms, _s, e, t);
    c.layers = this.layers, this.add(c);
  }
  /**
   * Must be called when the coordinate system of the cube camera is changed.
   */
  updateCoordinateSystem() {
    const e = this.coordinateSystem, t = this.children.concat(), [i, s, r, o, a, l] = t;
    for (const c of t) this.remove(c);
    if (e === Nn)
      i.up.set(0, 1, 0), i.lookAt(1, 0, 0), s.up.set(0, 1, 0), s.lookAt(-1, 0, 0), r.up.set(0, 0, -1), r.lookAt(0, 1, 0), o.up.set(0, 0, 1), o.lookAt(0, -1, 0), a.up.set(0, 1, 0), a.lookAt(0, 0, 1), l.up.set(0, 1, 0), l.lookAt(0, 0, -1);
    else if (e === zo)
      i.up.set(0, -1, 0), i.lookAt(-1, 0, 0), s.up.set(0, -1, 0), s.lookAt(1, 0, 0), r.up.set(0, 0, 1), r.lookAt(0, 1, 0), o.up.set(0, 0, -1), o.lookAt(0, -1, 0), a.up.set(0, -1, 0), a.lookAt(0, 0, 1), l.up.set(0, -1, 0), l.lookAt(0, 0, -1);
    else
      throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: " + e);
    for (const c of t)
      this.add(c), c.updateMatrixWorld();
  }
  /**
   * Calling this method will render the given scene with the given renderer
   * into the cube render target of the camera.
   *
   * @param {(Renderer|WebGLRenderer)} renderer - The renderer.
   * @param {Scene} scene - The scene to render.
   */
  update(e, t) {
    this.parent === null && this.updateMatrixWorld();
    const { renderTarget: i, activeMipmapLevel: s } = this;
    this.coordinateSystem !== e.coordinateSystem && (this.coordinateSystem = e.coordinateSystem, this.updateCoordinateSystem());
    const [r, o, a, l, c, u] = this.children, h = e.getRenderTarget(), f = e.getActiveCubeFace(), p = e.getActiveMipmapLevel(), v = e.xr.enabled;
    e.xr.enabled = !1;
    const x = i.texture.generateMipmaps;
    i.texture.generateMipmaps = !1, e.setRenderTarget(i, 0, s), e.render(t, r), e.setRenderTarget(i, 1, s), e.render(t, o), e.setRenderTarget(i, 2, s), e.render(t, a), e.setRenderTarget(i, 3, s), e.render(t, l), e.setRenderTarget(i, 4, s), e.render(t, c), i.texture.generateMipmaps = x, e.setRenderTarget(i, 5, s), e.render(t, u), e.setRenderTarget(h, f, p), e.xr.enabled = v, i.texture.needsPMREMUpdate = !0;
  }
}
class yd extends Zt {
  /**
   * Constructs a new cube texture.
   *
   * @param {Array<Image>} [images=[]] - An array holding a image for each side of a cube.
   * @param {number} [mapping=CubeReflectionMapping] - The texture mapping.
   * @param {number} [wrapS=ClampToEdgeWrapping] - The wrapS value.
   * @param {number} [wrapT=ClampToEdgeWrapping] - The wrapT value.
   * @param {number} [magFilter=LinearFilter] - The mag filter value.
   * @param {number} [minFilter=LinearMipmapLinearFilter] - The min filter value.
   * @param {number} [format=RGBAFormat] - The texture format.
   * @param {number} [type=UnsignedByteType] - The texture type.
   * @param {number} [anisotropy=Texture.DEFAULT_ANISOTROPY] - The anisotropy value.
   * @param {string} [colorSpace=NoColorSpace] - The color space value.
   */
  constructor(e = [], t = Fs, i, s, r, o, a, l, c, u) {
    super(e, t, i, s, r, o, a, l, c, u), this.isCubeTexture = !0, this.flipY = !1;
  }
  /**
   * Alias for {@link CubeTexture#image}.
   *
   * @type {Array<Image>}
   */
  get images() {
    return this.image;
  }
  set images(e) {
    this.image = e;
  }
}
class i0 extends ji {
  /**
   * Constructs a new cube render target.
   *
   * @param {number} [size=1] - The size of the render target.
   * @param {RenderTarget~Options} [options] - The configuration object.
   */
  constructor(e = 1, t = {}) {
    super(e, e, t), this.isWebGLCubeRenderTarget = !0;
    const i = { width: e, height: e, depth: 1 }, s = [i, i, i, i, i, i];
    this.texture = new yd(s), this._setTextureOptions(t), this.texture.isRenderTargetTexture = !0;
  }
  /**
   * Converts the given equirectangular texture to a cube map.
   *
   * @param {WebGLRenderer} renderer - The renderer.
   * @param {Texture} texture - The equirectangular texture.
   * @return {WebGLCubeRenderTarget} A reference to this cube render target.
   */
  fromEquirectangularTexture(e, t) {
    this.texture.type = t.type, this.texture.colorSpace = t.colorSpace, this.texture.generateMipmaps = t.generateMipmaps, this.texture.minFilter = t.minFilter, this.texture.magFilter = t.magFilter;
    const i = {
      uniforms: {
        tEquirect: { value: null }
      },
      vertexShader: (
        /* glsl */
        `

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`
      ),
      fragmentShader: (
        /* glsl */
        `

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`
      )
    }, s = new Ki(5, 5, 5), r = new Ei({
      name: "CubemapFromEquirect",
      uniforms: zs(i.uniforms),
      vertexShader: i.vertexShader,
      fragmentShader: i.fragmentShader,
      side: Wt,
      blending: xi
    });
    r.uniforms.tEquirect.value = t;
    const o = new vt(s, r), a = t.minFilter;
    return t.minFilter === ki && (t.minFilter = Un), new n0(1, 10, this).update(e, o), t.minFilter = a, o.geometry.dispose(), o.material.dispose(), this;
  }
  /**
   * Clears this cube render target.
   *
   * @param {WebGLRenderer} renderer - The renderer.
   * @param {boolean} [color=true] - Whether the color buffer should be cleared or not.
   * @param {boolean} [depth=true] - Whether the depth buffer should be cleared or not.
   * @param {boolean} [stencil=true] - Whether the stencil buffer should be cleared or not.
   */
  clear(e, t = !0, i = !0, s = !0) {
    const r = e.getRenderTarget();
    for (let o = 0; o < 6; o++)
      e.setRenderTarget(this, o), e.clear(t, i, s);
    e.setRenderTarget(r);
  }
}
class Dn extends Tt {
  constructor() {
    super(), this.isGroup = !0, this.type = "Group";
  }
}
const s0 = { type: "move" };
class Va {
  /**
   * Constructs a new XR controller.
   */
  constructor() {
    this._targetRay = null, this._grip = null, this._hand = null;
  }
  /**
   * Returns a group representing the hand space of the XR controller.
   *
   * @return {Group} A group representing the hand space of the XR controller.
   */
  getHandSpace() {
    return this._hand === null && (this._hand = new Dn(), this._hand.matrixAutoUpdate = !1, this._hand.visible = !1, this._hand.joints = {}, this._hand.inputState = { pinching: !1 }), this._hand;
  }
  /**
   * Returns a group representing the target ray space of the XR controller.
   *
   * @return {Group} A group representing the target ray space of the XR controller.
   */
  getTargetRaySpace() {
    return this._targetRay === null && (this._targetRay = new Dn(), this._targetRay.matrixAutoUpdate = !1, this._targetRay.visible = !1, this._targetRay.hasLinearVelocity = !1, this._targetRay.linearVelocity = new N(), this._targetRay.hasAngularVelocity = !1, this._targetRay.angularVelocity = new N()), this._targetRay;
  }
  /**
   * Returns a group representing the grip space of the XR controller.
   *
   * @return {Group} A group representing the grip space of the XR controller.
   */
  getGripSpace() {
    return this._grip === null && (this._grip = new Dn(), this._grip.matrixAutoUpdate = !1, this._grip.visible = !1, this._grip.hasLinearVelocity = !1, this._grip.linearVelocity = new N(), this._grip.hasAngularVelocity = !1, this._grip.angularVelocity = new N()), this._grip;
  }
  /**
   * Dispatches the given event to the groups representing
   * the different coordinate spaces of the XR controller.
   *
   * @param {Object} event - The event to dispatch.
   * @return {WebXRController} A reference to this instance.
   */
  dispatchEvent(e) {
    return this._targetRay !== null && this._targetRay.dispatchEvent(e), this._grip !== null && this._grip.dispatchEvent(e), this._hand !== null && this._hand.dispatchEvent(e), this;
  }
  /**
   * Connects the controller with the given XR input source.
   *
   * @param {XRInputSource} inputSource - The input source.
   * @return {WebXRController} A reference to this instance.
   */
  connect(e) {
    if (e && e.hand) {
      const t = this._hand;
      if (t)
        for (const i of e.hand.values())
          this._getHandJoint(t, i);
    }
    return this.dispatchEvent({ type: "connected", data: e }), this;
  }
  /**
   * Disconnects the controller from the given XR input source.
   *
   * @param {XRInputSource} inputSource - The input source.
   * @return {WebXRController} A reference to this instance.
   */
  disconnect(e) {
    return this.dispatchEvent({ type: "disconnected", data: e }), this._targetRay !== null && (this._targetRay.visible = !1), this._grip !== null && (this._grip.visible = !1), this._hand !== null && (this._hand.visible = !1), this;
  }
  /**
   * Updates the controller with the given input source, XR frame and reference space.
   * This updates the transformations of the groups that represent the different
   * coordinate systems of the controller.
   *
   * @param {XRInputSource} inputSource - The input source.
   * @param {XRFrame} frame - The XR frame.
   * @param {XRReferenceSpace} referenceSpace - The reference space.
   * @return {WebXRController} A reference to this instance.
   */
  update(e, t, i) {
    let s = null, r = null, o = null;
    const a = this._targetRay, l = this._grip, c = this._hand;
    if (e && t.session.visibilityState !== "visible-blurred") {
      if (c && e.hand) {
        o = !0;
        for (const x of e.hand.values()) {
          const m = t.getJointPose(x, i), d = this._getHandJoint(c, x);
          m !== null && (d.matrix.fromArray(m.transform.matrix), d.matrix.decompose(d.position, d.rotation, d.scale), d.matrixWorldNeedsUpdate = !0, d.jointRadius = m.radius), d.visible = m !== null;
        }
        const u = c.joints["index-finger-tip"], h = c.joints["thumb-tip"], f = u.position.distanceTo(h.position), p = 0.02, v = 5e-3;
        c.inputState.pinching && f > p + v ? (c.inputState.pinching = !1, this.dispatchEvent({
          type: "pinchend",
          handedness: e.handedness,
          target: this
        })) : !c.inputState.pinching && f <= p - v && (c.inputState.pinching = !0, this.dispatchEvent({
          type: "pinchstart",
          handedness: e.handedness,
          target: this
        }));
      } else
        l !== null && e.gripSpace && (r = t.getPose(e.gripSpace, i), r !== null && (l.matrix.fromArray(r.transform.matrix), l.matrix.decompose(l.position, l.rotation, l.scale), l.matrixWorldNeedsUpdate = !0, r.linearVelocity ? (l.hasLinearVelocity = !0, l.linearVelocity.copy(r.linearVelocity)) : l.hasLinearVelocity = !1, r.angularVelocity ? (l.hasAngularVelocity = !0, l.angularVelocity.copy(r.angularVelocity)) : l.hasAngularVelocity = !1));
      a !== null && (s = t.getPose(e.targetRaySpace, i), s === null && r !== null && (s = r), s !== null && (a.matrix.fromArray(s.transform.matrix), a.matrix.decompose(a.position, a.rotation, a.scale), a.matrixWorldNeedsUpdate = !0, s.linearVelocity ? (a.hasLinearVelocity = !0, a.linearVelocity.copy(s.linearVelocity)) : a.hasLinearVelocity = !1, s.angularVelocity ? (a.hasAngularVelocity = !0, a.angularVelocity.copy(s.angularVelocity)) : a.hasAngularVelocity = !1, this.dispatchEvent(s0)));
    }
    return a !== null && (a.visible = s !== null), l !== null && (l.visible = r !== null), c !== null && (c.visible = o !== null), this;
  }
  /**
   * Returns a group representing the hand joint for the given input joint.
   *
   * @private
   * @param {Group} hand - The group representing the hand space.
   * @param {XRJointSpace} inputjoint - The hand joint data.
   * @return {Group} A group representing the hand joint for the given input joint.
   */
  _getHandJoint(e, t) {
    if (e.joints[t.jointName] === void 0) {
      const i = new Dn();
      i.matrixAutoUpdate = !1, i.visible = !1, e.joints[t.jointName] = i, e.add(i);
    }
    return e.joints[t.jointName];
  }
}
class r0 extends Tt {
  /**
   * Constructs a new scene.
   */
  constructor() {
    super(), this.isScene = !0, this.type = "Scene", this.background = null, this.environment = null, this.fog = null, this.backgroundBlurriness = 0, this.backgroundIntensity = 1, this.backgroundRotation = new zn(), this.environmentIntensity = 1, this.environmentRotation = new zn(), this.overrideMaterial = null, typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe", { detail: this }));
  }
  copy(e, t) {
    return super.copy(e, t), e.background !== null && (this.background = e.background.clone()), e.environment !== null && (this.environment = e.environment.clone()), e.fog !== null && (this.fog = e.fog.clone()), this.backgroundBlurriness = e.backgroundBlurriness, this.backgroundIntensity = e.backgroundIntensity, this.backgroundRotation.copy(e.backgroundRotation), this.environmentIntensity = e.environmentIntensity, this.environmentRotation.copy(e.environmentRotation), e.overrideMaterial !== null && (this.overrideMaterial = e.overrideMaterial.clone()), this.matrixAutoUpdate = e.matrixAutoUpdate, this;
  }
  toJSON(e) {
    const t = super.toJSON(e);
    return this.fog !== null && (t.object.fog = this.fog.toJSON()), this.backgroundBlurriness > 0 && (t.object.backgroundBlurriness = this.backgroundBlurriness), this.backgroundIntensity !== 1 && (t.object.backgroundIntensity = this.backgroundIntensity), t.object.backgroundRotation = this.backgroundRotation.toArray(), this.environmentIntensity !== 1 && (t.object.environmentIntensity = this.environmentIntensity), t.object.environmentRotation = this.environmentRotation.toArray(), t;
  }
}
const ka = /* @__PURE__ */ new N(), o0 = /* @__PURE__ */ new N(), a0 = /* @__PURE__ */ new qe();
class mi {
  /**
   * Constructs a new plane.
   *
   * @param {Vector3} [normal=(1,0,0)] - A unit length vector defining the normal of the plane.
   * @param {number} [constant=0] - The signed distance from the origin to the plane.
   */
  constructor(e = new N(1, 0, 0), t = 0) {
    this.isPlane = !0, this.normal = e, this.constant = t;
  }
  /**
   * Sets the plane components by copying the given values.
   *
   * @param {Vector3} normal - The normal.
   * @param {number} constant - The constant.
   * @return {Plane} A reference to this plane.
   */
  set(e, t) {
    return this.normal.copy(e), this.constant = t, this;
  }
  /**
   * Sets the plane components by defining `x`, `y`, `z` as the
   * plane normal and `w` as the constant.
   *
   * @param {number} x - The value for the normal's x component.
   * @param {number} y - The value for the normal's y component.
   * @param {number} z - The value for the normal's z component.
   * @param {number} w - The constant value.
   * @return {Plane} A reference to this plane.
   */
  setComponents(e, t, i, s) {
    return this.normal.set(e, t, i), this.constant = s, this;
  }
  /**
   * Sets the plane from the given normal and coplanar point (that is a point
   * that lies onto the plane).
   *
   * @param {Vector3} normal - The normal.
   * @param {Vector3} point - A coplanar point.
   * @return {Plane} A reference to this plane.
   */
  setFromNormalAndCoplanarPoint(e, t) {
    return this.normal.copy(e), this.constant = -t.dot(this.normal), this;
  }
  /**
   * Sets the plane from three coplanar points. The winding order is
   * assumed to be counter-clockwise, and determines the direction of
   * the plane normal.
   *
   * @param {Vector3} a - The first coplanar point.
   * @param {Vector3} b - The second coplanar point.
   * @param {Vector3} c - The third coplanar point.
   * @return {Plane} A reference to this plane.
   */
  setFromCoplanarPoints(e, t, i) {
    const s = ka.subVectors(i, t).cross(o0.subVectors(e, t)).normalize();
    return this.setFromNormalAndCoplanarPoint(s, e), this;
  }
  /**
   * Copies the values of the given plane to this instance.
   *
   * @param {Plane} plane - The plane to copy.
   * @return {Plane} A reference to this plane.
   */
  copy(e) {
    return this.normal.copy(e.normal), this.constant = e.constant, this;
  }
  /**
   * Normalizes the plane normal and adjusts the constant accordingly.
   *
   * @return {Plane} A reference to this plane.
   */
  normalize() {
    const e = 1 / this.normal.length();
    return this.normal.multiplyScalar(e), this.constant *= e, this;
  }
  /**
   * Negates both the plane normal and the constant.
   *
   * @return {Plane} A reference to this plane.
   */
  negate() {
    return this.constant *= -1, this.normal.negate(), this;
  }
  /**
   * Returns the signed distance from the given point to this plane.
   *
   * @param {Vector3} point - The point to compute the distance for.
   * @return {number} The signed distance.
   */
  distanceToPoint(e) {
    return this.normal.dot(e) + this.constant;
  }
  /**
   * Returns the signed distance from the given sphere to this plane.
   *
   * @param {Sphere} sphere - The sphere to compute the distance for.
   * @return {number} The signed distance.
   */
  distanceToSphere(e) {
    return this.distanceToPoint(e.center) - e.radius;
  }
  /**
   * Projects a the given point onto the plane.
   *
   * @param {Vector3} point - The point to project.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The projected point on the plane.
   */
  projectPoint(e, t) {
    return t.copy(e).addScaledVector(this.normal, -this.distanceToPoint(e));
  }
  /**
   * Returns the intersection point of the passed line and the plane. Returns
   * `null` if the line does not intersect. Returns the line's starting point if
   * the line is coplanar with the plane.
   *
   * @param {Line3} line - The line to compute the intersection for.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The intersection point.
   */
  intersectLine(e, t) {
    const i = e.delta(ka), s = this.normal.dot(i);
    if (s === 0)
      return this.distanceToPoint(e.start) === 0 ? t.copy(e.start) : null;
    const r = -(e.start.dot(this.normal) + this.constant) / s;
    return r < 0 || r > 1 ? null : t.copy(e.start).addScaledVector(i, r);
  }
  /**
   * Returns `true` if the given line segment intersects with (passes through) the plane.
   *
   * @param {Line3} line - The line to test.
   * @return {boolean} Whether the given line segment intersects with the plane or not.
   */
  intersectsLine(e) {
    const t = this.distanceToPoint(e.start), i = this.distanceToPoint(e.end);
    return t < 0 && i > 0 || i < 0 && t > 0;
  }
  /**
   * Returns `true` if the given bounding box intersects with the plane.
   *
   * @param {Box3} box - The bounding box to test.
   * @return {boolean} Whether the given bounding box intersects with the plane or not.
   */
  intersectsBox(e) {
    return e.intersectsPlane(this);
  }
  /**
   * Returns `true` if the given bounding sphere intersects with the plane.
   *
   * @param {Sphere} sphere - The bounding sphere to test.
   * @return {boolean} Whether the given bounding sphere intersects with the plane or not.
   */
  intersectsSphere(e) {
    return e.intersectsPlane(this);
  }
  /**
   * Returns a coplanar vector to the plane, by calculating the
   * projection of the normal at the origin onto the plane.
   *
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The coplanar point.
   */
  coplanarPoint(e) {
    return e.copy(this.normal).multiplyScalar(-this.constant);
  }
  /**
   * Apply a 4x4 matrix to the plane. The matrix must be an affine, homogeneous transform.
   *
   * The optional normal matrix can be pre-computed like so:
   * ```js
   * const optionalNormalMatrix = new THREE.Matrix3().getNormalMatrix( matrix );
   * ```
   *
   * @param {Matrix4} matrix - The transformation matrix.
   * @param {Matrix4} [optionalNormalMatrix] - A pre-computed normal matrix.
   * @return {Plane} A reference to this plane.
   */
  applyMatrix4(e, t) {
    const i = t || a0.getNormalMatrix(e), s = this.coplanarPoint(ka).applyMatrix4(e), r = this.normal.applyMatrix3(i).normalize();
    return this.constant = -s.dot(r), this;
  }
  /**
   * Translates the plane by the distance defined by the given offset vector.
   * Note that this only affects the plane constant and will not affect the normal vector.
   *
   * @param {Vector3} offset - The offset vector.
   * @return {Plane} A reference to this plane.
   */
  translate(e) {
    return this.constant -= e.dot(this.normal), this;
  }
  /**
   * Returns `true` if this plane is equal with the given one.
   *
   * @param {Plane} plane - The plane to test for equality.
   * @return {boolean} Whether this plane is equal with the given one.
   */
  equals(e) {
    return e.normal.equals(this.normal) && e.constant === this.constant;
  }
  /**
   * Returns a new plane with copied values from this instance.
   *
   * @return {Plane} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
}
const Ii = /* @__PURE__ */ new Fr(), l0 = /* @__PURE__ */ new Ve(0.5, 0.5), ro = /* @__PURE__ */ new N();
class wc {
  /**
   * Constructs a new frustum.
   *
   * @param {Plane} [p0] - The first plane that encloses the frustum.
   * @param {Plane} [p1] - The second plane that encloses the frustum.
   * @param {Plane} [p2] - The third plane that encloses the frustum.
   * @param {Plane} [p3] - The fourth plane that encloses the frustum.
   * @param {Plane} [p4] - The fifth plane that encloses the frustum.
   * @param {Plane} [p5] - The sixth plane that encloses the frustum.
   */
  constructor(e = new mi(), t = new mi(), i = new mi(), s = new mi(), r = new mi(), o = new mi()) {
    this.planes = [e, t, i, s, r, o];
  }
  /**
   * Sets the frustum planes by copying the given planes.
   *
   * @param {Plane} [p0] - The first plane that encloses the frustum.
   * @param {Plane} [p1] - The second plane that encloses the frustum.
   * @param {Plane} [p2] - The third plane that encloses the frustum.
   * @param {Plane} [p3] - The fourth plane that encloses the frustum.
   * @param {Plane} [p4] - The fifth plane that encloses the frustum.
   * @param {Plane} [p5] - The sixth plane that encloses the frustum.
   * @return {Frustum} A reference to this frustum.
   */
  set(e, t, i, s, r, o) {
    const a = this.planes;
    return a[0].copy(e), a[1].copy(t), a[2].copy(i), a[3].copy(s), a[4].copy(r), a[5].copy(o), this;
  }
  /**
   * Copies the values of the given frustum to this instance.
   *
   * @param {Frustum} frustum - The frustum to copy.
   * @return {Frustum} A reference to this frustum.
   */
  copy(e) {
    const t = this.planes;
    for (let i = 0; i < 6; i++)
      t[i].copy(e.planes[i]);
    return this;
  }
  /**
   * Sets the frustum planes from the given projection matrix.
   *
   * @param {Matrix4} m - The projection matrix.
   * @param {(WebGLCoordinateSystem|WebGPUCoordinateSystem)} coordinateSystem - The coordinate system.
   * @param {boolean} [reversedDepth=false] - Whether to use a reversed depth.
   * @return {Frustum} A reference to this frustum.
   */
  setFromProjectionMatrix(e, t = Nn, i = !1) {
    const s = this.planes, r = e.elements, o = r[0], a = r[1], l = r[2], c = r[3], u = r[4], h = r[5], f = r[6], p = r[7], v = r[8], x = r[9], m = r[10], d = r[11], b = r[12], A = r[13], M = r[14], C = r[15];
    if (s[0].setComponents(c - o, p - u, d - v, C - b).normalize(), s[1].setComponents(c + o, p + u, d + v, C + b).normalize(), s[2].setComponents(c + a, p + h, d + x, C + A).normalize(), s[3].setComponents(c - a, p - h, d - x, C - A).normalize(), i)
      s[4].setComponents(l, f, m, M).normalize(), s[5].setComponents(c - l, p - f, d - m, C - M).normalize();
    else if (s[4].setComponents(c - l, p - f, d - m, C - M).normalize(), t === Nn)
      s[5].setComponents(c + l, p + f, d + m, C + M).normalize();
    else if (t === zo)
      s[5].setComponents(l, f, m, M).normalize();
    else
      throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: " + t);
    return this;
  }
  /**
   * Returns `true` if the 3D object's bounding sphere is intersecting this frustum.
   *
   * Note that the 3D object must have a geometry so that the bounding sphere can be calculated.
   *
   * @param {Object3D} object - The 3D object to test.
   * @return {boolean} Whether the 3D object's bounding sphere is intersecting this frustum or not.
   */
  intersectsObject(e) {
    if (e.boundingSphere !== void 0)
      e.boundingSphere === null && e.computeBoundingSphere(), Ii.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);
    else {
      const t = e.geometry;
      t.boundingSphere === null && t.computeBoundingSphere(), Ii.copy(t.boundingSphere).applyMatrix4(e.matrixWorld);
    }
    return this.intersectsSphere(Ii);
  }
  /**
   * Returns `true` if the given sprite is intersecting this frustum.
   *
   * @param {Sprite} sprite - The sprite to test.
   * @return {boolean} Whether the sprite is intersecting this frustum or not.
   */
  intersectsSprite(e) {
    Ii.center.set(0, 0, 0);
    const t = l0.distanceTo(e.center);
    return Ii.radius = 0.7071067811865476 + t, Ii.applyMatrix4(e.matrixWorld), this.intersectsSphere(Ii);
  }
  /**
   * Returns `true` if the given bounding sphere is intersecting this frustum.
   *
   * @param {Sphere} sphere - The bounding sphere to test.
   * @return {boolean} Whether the bounding sphere is intersecting this frustum or not.
   */
  intersectsSphere(e) {
    const t = this.planes, i = e.center, s = -e.radius;
    for (let r = 0; r < 6; r++)
      if (t[r].distanceToPoint(i) < s)
        return !1;
    return !0;
  }
  /**
   * Returns `true` if the given bounding box is intersecting this frustum.
   *
   * @param {Box3} box - The bounding box to test.
   * @return {boolean} Whether the bounding box is intersecting this frustum or not.
   */
  intersectsBox(e) {
    const t = this.planes;
    for (let i = 0; i < 6; i++) {
      const s = t[i];
      if (ro.x = s.normal.x > 0 ? e.max.x : e.min.x, ro.y = s.normal.y > 0 ? e.max.y : e.min.y, ro.z = s.normal.z > 0 ? e.max.z : e.min.z, s.distanceToPoint(ro) < 0)
        return !1;
    }
    return !0;
  }
  /**
   * Returns `true` if the given point lies within the frustum.
   *
   * @param {Vector3} point - The point to test.
   * @return {boolean} Whether the point lies within this frustum or not.
   */
  containsPoint(e) {
    const t = this.planes;
    for (let i = 0; i < 6; i++)
      if (t[i].distanceToPoint(e) < 0)
        return !1;
    return !0;
  }
  /**
   * Returns a new frustum with copied values from this instance.
   *
   * @return {Frustum} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
}
class Rc extends Qi {
  /**
   * Constructs a new line basic material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isLineBasicMaterial = !0, this.type = "LineBasicMaterial", this.color = new Xe(16777215), this.map = null, this.linewidth = 1, this.linecap = "round", this.linejoin = "round", this.fog = !0, this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.color.copy(e.color), this.map = e.map, this.linewidth = e.linewidth, this.linecap = e.linecap, this.linejoin = e.linejoin, this.fog = e.fog, this;
  }
}
const Vo = /* @__PURE__ */ new N(), ko = /* @__PURE__ */ new N(), $u = /* @__PURE__ */ new pt(), er = /* @__PURE__ */ new na(), oo = /* @__PURE__ */ new Fr(), Ga = /* @__PURE__ */ new N(), Zu = /* @__PURE__ */ new N();
class c0 extends Tt {
  /**
   * Constructs a new line.
   *
   * @param {BufferGeometry} [geometry] - The line geometry.
   * @param {Material|Array<Material>} [material] - The line material.
   */
  constructor(e = new Nt(), t = new Rc()) {
    super(), this.isLine = !0, this.type = "Line", this.geometry = e, this.material = t, this.morphTargetDictionary = void 0, this.morphTargetInfluences = void 0, this.updateMorphTargets();
  }
  copy(e, t) {
    return super.copy(e, t), this.material = Array.isArray(e.material) ? e.material.slice() : e.material, this.geometry = e.geometry, this;
  }
  /**
   * Computes an array of distance values which are necessary for rendering dashed lines.
   * For each vertex in the geometry, the method calculates the cumulative length from the
   * current point to the very beginning of the line.
   *
   * @return {Line} A reference to this line.
   */
  computeLineDistances() {
    const e = this.geometry;
    if (e.index === null) {
      const t = e.attributes.position, i = [0];
      for (let s = 1, r = t.count; s < r; s++)
        Vo.fromBufferAttribute(t, s - 1), ko.fromBufferAttribute(t, s), i[s] = i[s - 1], i[s] += Vo.distanceTo(ko);
      e.setAttribute("lineDistance", new mt(i, 1));
    } else
      console.warn("THREE.Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");
    return this;
  }
  /**
   * Computes intersection points between a casted ray and this line.
   *
   * @param {Raycaster} raycaster - The raycaster.
   * @param {Array<Object>} intersects - The target array that holds the intersection points.
   */
  raycast(e, t) {
    const i = this.geometry, s = this.matrixWorld, r = e.params.Line.threshold, o = i.drawRange;
    if (i.boundingSphere === null && i.computeBoundingSphere(), oo.copy(i.boundingSphere), oo.applyMatrix4(s), oo.radius += r, e.ray.intersectsSphere(oo) === !1) return;
    $u.copy(s).invert(), er.copy(e.ray).applyMatrix4($u);
    const a = r / ((this.scale.x + this.scale.y + this.scale.z) / 3), l = a * a, c = this.isLineSegments ? 2 : 1, u = i.index, f = i.attributes.position;
    if (u !== null) {
      const p = Math.max(0, o.start), v = Math.min(u.count, o.start + o.count);
      for (let x = p, m = v - 1; x < m; x += c) {
        const d = u.getX(x), b = u.getX(x + 1), A = ao(this, e, er, l, d, b, x);
        A && t.push(A);
      }
      if (this.isLineLoop) {
        const x = u.getX(v - 1), m = u.getX(p), d = ao(this, e, er, l, x, m, v - 1);
        d && t.push(d);
      }
    } else {
      const p = Math.max(0, o.start), v = Math.min(f.count, o.start + o.count);
      for (let x = p, m = v - 1; x < m; x += c) {
        const d = ao(this, e, er, l, x, x + 1, x);
        d && t.push(d);
      }
      if (this.isLineLoop) {
        const x = ao(this, e, er, l, v - 1, p, v - 1);
        x && t.push(x);
      }
    }
  }
  /**
   * Sets the values of {@link Line#morphTargetDictionary} and {@link Line#morphTargetInfluences}
   * to make sure existing morph targets can influence this 3D object.
   */
  updateMorphTargets() {
    const t = this.geometry.morphAttributes, i = Object.keys(t);
    if (i.length > 0) {
      const s = t[i[0]];
      if (s !== void 0) {
        this.morphTargetInfluences = [], this.morphTargetDictionary = {};
        for (let r = 0, o = s.length; r < o; r++) {
          const a = s[r].name || String(r);
          this.morphTargetInfluences.push(0), this.morphTargetDictionary[a] = r;
        }
      }
    }
  }
}
function ao(n, e, t, i, s, r, o) {
  const a = n.geometry.attributes.position;
  if (Vo.fromBufferAttribute(a, s), ko.fromBufferAttribute(a, r), t.distanceSqToSegment(Vo, ko, Ga, Zu) > i) return;
  Ga.applyMatrix4(n.matrixWorld);
  const c = e.ray.origin.distanceTo(Ga);
  if (!(c < e.near || c > e.far))
    return {
      distance: c,
      // What do we want? intersection point on the ray or on the segment??
      // point: raycaster.ray.at( distance ),
      point: Zu.clone().applyMatrix4(n.matrixWorld),
      index: o,
      face: null,
      faceIndex: null,
      barycoord: null,
      object: n
    };
}
const Ju = /* @__PURE__ */ new N(), Qu = /* @__PURE__ */ new N();
class Ed extends c0 {
  /**
   * Constructs a new line segments.
   *
   * @param {BufferGeometry} [geometry] - The line geometry.
   * @param {Material|Array<Material>} [material] - The line material.
   */
  constructor(e, t) {
    super(e, t), this.isLineSegments = !0, this.type = "LineSegments";
  }
  computeLineDistances() {
    const e = this.geometry;
    if (e.index === null) {
      const t = e.attributes.position, i = [];
      for (let s = 0, r = t.count; s < r; s += 2)
        Ju.fromBufferAttribute(t, s), Qu.fromBufferAttribute(t, s + 1), i[s] = s === 0 ? 0 : i[s - 1], i[s + 1] = i[s] + Ju.distanceTo(Qu);
      e.setAttribute("lineDistance", new mt(i, 1));
    } else
      console.warn("THREE.LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");
    return this;
  }
}
class Td extends Qi {
  /**
   * Constructs a new points material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isPointsMaterial = !0, this.type = "PointsMaterial", this.color = new Xe(16777215), this.map = null, this.alphaMap = null, this.size = 1, this.sizeAttenuation = !0, this.fog = !0, this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.color.copy(e.color), this.map = e.map, this.alphaMap = e.alphaMap, this.size = e.size, this.sizeAttenuation = e.sizeAttenuation, this.fog = e.fog, this;
  }
}
const eh = /* @__PURE__ */ new pt(), ec = /* @__PURE__ */ new na(), lo = /* @__PURE__ */ new Fr(), co = /* @__PURE__ */ new N();
class u0 extends Tt {
  /**
   * Constructs a new point cloud.
   *
   * @param {BufferGeometry} [geometry] - The points geometry.
   * @param {Material|Array<Material>} [material] - The points material.
   */
  constructor(e = new Nt(), t = new Td()) {
    super(), this.isPoints = !0, this.type = "Points", this.geometry = e, this.material = t, this.morphTargetDictionary = void 0, this.morphTargetInfluences = void 0, this.updateMorphTargets();
  }
  copy(e, t) {
    return super.copy(e, t), this.material = Array.isArray(e.material) ? e.material.slice() : e.material, this.geometry = e.geometry, this;
  }
  /**
   * Computes intersection points between a casted ray and this point cloud.
   *
   * @param {Raycaster} raycaster - The raycaster.
   * @param {Array<Object>} intersects - The target array that holds the intersection points.
   */
  raycast(e, t) {
    const i = this.geometry, s = this.matrixWorld, r = e.params.Points.threshold, o = i.drawRange;
    if (i.boundingSphere === null && i.computeBoundingSphere(), lo.copy(i.boundingSphere), lo.applyMatrix4(s), lo.radius += r, e.ray.intersectsSphere(lo) === !1) return;
    eh.copy(s).invert(), ec.copy(e.ray).applyMatrix4(eh);
    const a = r / ((this.scale.x + this.scale.y + this.scale.z) / 3), l = a * a, c = i.index, h = i.attributes.position;
    if (c !== null) {
      const f = Math.max(0, o.start), p = Math.min(c.count, o.start + o.count);
      for (let v = f, x = p; v < x; v++) {
        const m = c.getX(v);
        co.fromBufferAttribute(h, m), th(co, m, l, s, e, t, this);
      }
    } else {
      const f = Math.max(0, o.start), p = Math.min(h.count, o.start + o.count);
      for (let v = f, x = p; v < x; v++)
        co.fromBufferAttribute(h, v), th(co, v, l, s, e, t, this);
    }
  }
  /**
   * Sets the values of {@link Points#morphTargetDictionary} and {@link Points#morphTargetInfluences}
   * to make sure existing morph targets can influence this 3D object.
   */
  updateMorphTargets() {
    const t = this.geometry.morphAttributes, i = Object.keys(t);
    if (i.length > 0) {
      const s = t[i[0]];
      if (s !== void 0) {
        this.morphTargetInfluences = [], this.morphTargetDictionary = {};
        for (let r = 0, o = s.length; r < o; r++) {
          const a = s[r].name || String(r);
          this.morphTargetInfluences.push(0), this.morphTargetDictionary[a] = r;
        }
      }
    }
  }
}
function th(n, e, t, i, s, r, o) {
  const a = ec.distanceSqToPoint(n);
  if (a < t) {
    const l = new N();
    ec.closestPointToPoint(n, l), l.applyMatrix4(i);
    const c = s.ray.origin.distanceTo(l);
    if (c < s.near || c > s.far) return;
    r.push({
      distance: c,
      distanceToRay: Math.sqrt(a),
      point: l,
      index: e,
      face: null,
      faceIndex: null,
      barycoord: null,
      object: o
    });
  }
}
class bd extends Zt {
  /**
   * Constructs a new depth texture.
   *
   * @param {number} width - The width of the texture.
   * @param {number} height - The height of the texture.
   * @param {number} [type=UnsignedIntType] - The texture type.
   * @param {number} [mapping=Texture.DEFAULT_MAPPING] - The texture mapping.
   * @param {number} [wrapS=ClampToEdgeWrapping] - The wrapS value.
   * @param {number} [wrapT=ClampToEdgeWrapping] - The wrapT value.
   * @param {number} [magFilter=LinearFilter] - The mag filter value.
   * @param {number} [minFilter=LinearFilter] - The min filter value.
   * @param {number} [anisotropy=Texture.DEFAULT_ANISOTROPY] - The anisotropy value.
   * @param {number} [format=DepthFormat] - The texture format.
   * @param {number} [depth=1] - The depth of the texture.
   */
  constructor(e, t, i = Yi, s, r, o, a = yn, l = yn, c, u = wr, h = 1) {
    if (u !== wr && u !== Rr)
      throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");
    const f = { width: e, height: t, depth: h };
    super(f, s, r, o, a, l, u, i, c), this.isDepthTexture = !0, this.flipY = !1, this.generateMipmaps = !1, this.compareFunction = null;
  }
  copy(e) {
    return super.copy(e), this.source = new Ac(Object.assign({}, e.image)), this.compareFunction = e.compareFunction, this;
  }
  toJSON(e) {
    const t = super.toJSON(e);
    return this.compareFunction !== null && (t.compareFunction = this.compareFunction), t;
  }
}
class Ad extends Zt {
  /**
   * Creates a new raw texture.
   *
   * @param {?(WebGLTexture|GPUTexture)} [sourceTexture=null] - The external texture.
   */
  constructor(e = null) {
    super(), this.sourceTexture = e, this.isExternalTexture = !0;
  }
  copy(e) {
    return super.copy(e), this.sourceTexture = e.sourceTexture, this;
  }
}
class Cc extends Nt {
  /**
   * Constructs a new capsule geometry.
   *
   * @param {number} [radius=1] - Radius of the capsule.
   * @param {number} [height=1] - Height of the middle section.
   * @param {number} [capSegments=4] - Number of curve segments used to build each cap.
   * @param {number} [radialSegments=8] - Number of segmented faces around the circumference of the capsule. Must be an integer >= 3.
   * @param {number} [heightSegments=1] - Number of rows of faces along the height of the middle section. Must be an integer >= 1.
   */
  constructor(e = 1, t = 1, i = 4, s = 8, r = 1) {
    super(), this.type = "CapsuleGeometry", this.parameters = {
      radius: e,
      height: t,
      capSegments: i,
      radialSegments: s,
      heightSegments: r
    }, t = Math.max(0, t), i = Math.max(1, Math.floor(i)), s = Math.max(3, Math.floor(s)), r = Math.max(1, Math.floor(r));
    const o = [], a = [], l = [], c = [], u = t / 2, h = Math.PI / 2 * e, f = t, p = 2 * h + f, v = i * 2 + r, x = s + 1, m = new N(), d = new N();
    for (let b = 0; b <= v; b++) {
      let A = 0, M = 0, C = 0, w = 0;
      if (b <= i) {
        const S = b / i, y = S * Math.PI / 2;
        M = -u - e * Math.cos(y), C = e * Math.sin(y), w = -e * Math.cos(y), A = S * h;
      } else if (b <= i + r) {
        const S = (b - i) / r;
        M = -u + S * t, C = e, w = 0, A = h + S * f;
      } else {
        const S = (b - i - r) / i, y = S * Math.PI / 2;
        M = u + e * Math.sin(y), C = e * Math.cos(y), w = e * Math.sin(y), A = h + f + S * h;
      }
      const P = Math.max(0, Math.min(1, A / p));
      let U = 0;
      b === 0 ? U = 0.5 / s : b === v && (U = -0.5 / s);
      for (let S = 0; S <= s; S++) {
        const y = S / s, D = y * Math.PI * 2, L = Math.sin(D), V = Math.cos(D);
        d.x = -C * V, d.y = M, d.z = C * L, a.push(d.x, d.y, d.z), m.set(
          -C * V,
          w,
          C * L
        ), m.normalize(), l.push(m.x, m.y, m.z), c.push(y + U, P);
      }
      if (b > 0) {
        const S = (b - 1) * x;
        for (let y = 0; y < s; y++) {
          const D = S + y, L = S + y + 1, V = b * x + y, Z = b * x + y + 1;
          o.push(D, L, V), o.push(L, Z, V);
        }
      }
    }
    this.setIndex(o), this.setAttribute("position", new mt(a, 3)), this.setAttribute("normal", new mt(l, 3)), this.setAttribute("uv", new mt(c, 2));
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {CapsuleGeometry} A new instance.
   */
  static fromJSON(e) {
    return new Cc(e.radius, e.height, e.capSegments, e.radialSegments, e.heightSegments);
  }
}
class Pc extends Nt {
  /**
   * Constructs a new polyhedron geometry.
   *
   * @param {Array<number>} [vertices] - A flat array of vertices describing the base shape.
   * @param {Array<number>} [indices] - A flat array of indices describing the base shape.
   * @param {number} [radius=1] - The radius of the shape.
   * @param {number} [detail=0] - How many levels to subdivide the geometry. The more detail, the smoother the shape.
   */
  constructor(e = [], t = [], i = 1, s = 0) {
    super(), this.type = "PolyhedronGeometry", this.parameters = {
      vertices: e,
      indices: t,
      radius: i,
      detail: s
    };
    const r = [], o = [];
    a(s), c(i), u(), this.setAttribute("position", new mt(r, 3)), this.setAttribute("normal", new mt(r.slice(), 3)), this.setAttribute("uv", new mt(o, 2)), s === 0 ? this.computeVertexNormals() : this.normalizeNormals();
    function a(b) {
      const A = new N(), M = new N(), C = new N();
      for (let w = 0; w < t.length; w += 3)
        p(t[w + 0], A), p(t[w + 1], M), p(t[w + 2], C), l(A, M, C, b);
    }
    function l(b, A, M, C) {
      const w = C + 1, P = [];
      for (let U = 0; U <= w; U++) {
        P[U] = [];
        const S = b.clone().lerp(M, U / w), y = A.clone().lerp(M, U / w), D = w - U;
        for (let L = 0; L <= D; L++)
          L === 0 && U === w ? P[U][L] = S : P[U][L] = S.clone().lerp(y, L / D);
      }
      for (let U = 0; U < w; U++)
        for (let S = 0; S < 2 * (w - U) - 1; S++) {
          const y = Math.floor(S / 2);
          S % 2 === 0 ? (f(P[U][y + 1]), f(P[U + 1][y]), f(P[U][y])) : (f(P[U][y + 1]), f(P[U + 1][y + 1]), f(P[U + 1][y]));
        }
    }
    function c(b) {
      const A = new N();
      for (let M = 0; M < r.length; M += 3)
        A.x = r[M + 0], A.y = r[M + 1], A.z = r[M + 2], A.normalize().multiplyScalar(b), r[M + 0] = A.x, r[M + 1] = A.y, r[M + 2] = A.z;
    }
    function u() {
      const b = new N();
      for (let A = 0; A < r.length; A += 3) {
        b.x = r[A + 0], b.y = r[A + 1], b.z = r[A + 2];
        const M = m(b) / 2 / Math.PI + 0.5, C = d(b) / Math.PI + 0.5;
        o.push(M, 1 - C);
      }
      v(), h();
    }
    function h() {
      for (let b = 0; b < o.length; b += 6) {
        const A = o[b + 0], M = o[b + 2], C = o[b + 4], w = Math.max(A, M, C), P = Math.min(A, M, C);
        w > 0.9 && P < 0.1 && (A < 0.2 && (o[b + 0] += 1), M < 0.2 && (o[b + 2] += 1), C < 0.2 && (o[b + 4] += 1));
      }
    }
    function f(b) {
      r.push(b.x, b.y, b.z);
    }
    function p(b, A) {
      const M = b * 3;
      A.x = e[M + 0], A.y = e[M + 1], A.z = e[M + 2];
    }
    function v() {
      const b = new N(), A = new N(), M = new N(), C = new N(), w = new Ve(), P = new Ve(), U = new Ve();
      for (let S = 0, y = 0; S < r.length; S += 9, y += 6) {
        b.set(r[S + 0], r[S + 1], r[S + 2]), A.set(r[S + 3], r[S + 4], r[S + 5]), M.set(r[S + 6], r[S + 7], r[S + 8]), w.set(o[y + 0], o[y + 1]), P.set(o[y + 2], o[y + 3]), U.set(o[y + 4], o[y + 5]), C.copy(b).add(A).add(M).divideScalar(3);
        const D = m(C);
        x(w, y + 0, b, D), x(P, y + 2, A, D), x(U, y + 4, M, D);
      }
    }
    function x(b, A, M, C) {
      C < 0 && b.x === 1 && (o[A] = b.x - 1), M.x === 0 && M.z === 0 && (o[A] = C / 2 / Math.PI + 0.5);
    }
    function m(b) {
      return Math.atan2(b.z, -b.x);
    }
    function d(b) {
      return Math.atan2(-b.y, Math.sqrt(b.x * b.x + b.z * b.z));
    }
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {PolyhedronGeometry} A new instance.
   */
  static fromJSON(e) {
    return new Pc(e.vertices, e.indices, e.radius, e.details);
  }
}
const uo = /* @__PURE__ */ new N(), ho = /* @__PURE__ */ new N(), Wa = /* @__PURE__ */ new N(), fo = /* @__PURE__ */ new fn();
class h0 extends Nt {
  /**
   * Constructs a new edges geometry.
   *
   * @param {?BufferGeometry} [geometry=null] - The geometry.
   * @param {number} [thresholdAngle=1] - An edge is only rendered if the angle (in degrees)
   * between the face normals of the adjoining faces exceeds this value.
   */
  constructor(e = null, t = 1) {
    if (super(), this.type = "EdgesGeometry", this.parameters = {
      geometry: e,
      thresholdAngle: t
    }, e !== null) {
      const s = Math.pow(10, 4), r = Math.cos(pr * t), o = e.getIndex(), a = e.getAttribute("position"), l = o ? o.count : a.count, c = [0, 0, 0], u = ["a", "b", "c"], h = new Array(3), f = {}, p = [];
      for (let v = 0; v < l; v += 3) {
        o ? (c[0] = o.getX(v), c[1] = o.getX(v + 1), c[2] = o.getX(v + 2)) : (c[0] = v, c[1] = v + 1, c[2] = v + 2);
        const { a: x, b: m, c: d } = fo;
        if (x.fromBufferAttribute(a, c[0]), m.fromBufferAttribute(a, c[1]), d.fromBufferAttribute(a, c[2]), fo.getNormal(Wa), h[0] = `${Math.round(x.x * s)},${Math.round(x.y * s)},${Math.round(x.z * s)}`, h[1] = `${Math.round(m.x * s)},${Math.round(m.y * s)},${Math.round(m.z * s)}`, h[2] = `${Math.round(d.x * s)},${Math.round(d.y * s)},${Math.round(d.z * s)}`, !(h[0] === h[1] || h[1] === h[2] || h[2] === h[0]))
          for (let b = 0; b < 3; b++) {
            const A = (b + 1) % 3, M = h[b], C = h[A], w = fo[u[b]], P = fo[u[A]], U = `${M}_${C}`, S = `${C}_${M}`;
            S in f && f[S] ? (Wa.dot(f[S].normal) <= r && (p.push(w.x, w.y, w.z), p.push(P.x, P.y, P.z)), f[S] = null) : U in f || (f[U] = {
              index0: c[b],
              index1: c[A],
              normal: Wa.clone()
            });
          }
      }
      for (const v in f)
        if (f[v]) {
          const { index0: x, index1: m } = f[v];
          uo.fromBufferAttribute(a, x), ho.fromBufferAttribute(a, m), p.push(uo.x, uo.y, uo.z), p.push(ho.x, ho.y, ho.z);
        }
      this.setAttribute("position", new mt(p, 3));
    }
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
}
class Dc extends Pc {
  /**
   * Constructs a new icosahedron geometry.
   *
   * @param {number} [radius=1] - Radius of the icosahedron.
   * @param {number} [detail=0] - Setting this to a value greater than `0` adds vertices making it no longer a icosahedron.
   */
  constructor(e = 1, t = 0) {
    const i = (1 + Math.sqrt(5)) / 2, s = [
      -1,
      i,
      0,
      1,
      i,
      0,
      -1,
      -i,
      0,
      1,
      -i,
      0,
      0,
      -1,
      i,
      0,
      1,
      i,
      0,
      -1,
      -i,
      0,
      1,
      -i,
      i,
      0,
      -1,
      i,
      0,
      1,
      -i,
      0,
      -1,
      -i,
      0,
      1
    ], r = [
      0,
      11,
      5,
      0,
      5,
      1,
      0,
      1,
      7,
      0,
      7,
      10,
      0,
      10,
      11,
      1,
      5,
      9,
      5,
      11,
      4,
      11,
      10,
      2,
      10,
      7,
      6,
      7,
      1,
      8,
      3,
      9,
      4,
      3,
      4,
      2,
      3,
      2,
      6,
      3,
      6,
      8,
      3,
      8,
      9,
      4,
      9,
      5,
      2,
      4,
      11,
      6,
      2,
      10,
      8,
      6,
      7,
      9,
      8,
      1
    ];
    super(s, r, e, t), this.type = "IcosahedronGeometry", this.parameters = {
      radius: e,
      detail: t
    };
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {IcosahedronGeometry} A new instance.
   */
  static fromJSON(e) {
    return new Dc(e.radius, e.detail);
  }
}
class Hs extends Nt {
  /**
   * Constructs a new plane geometry.
   *
   * @param {number} [width=1] - The width along the X axis.
   * @param {number} [height=1] - The height along the Y axis
   * @param {number} [widthSegments=1] - The number of segments along the X axis.
   * @param {number} [heightSegments=1] - The number of segments along the Y axis.
   */
  constructor(e = 1, t = 1, i = 1, s = 1) {
    super(), this.type = "PlaneGeometry", this.parameters = {
      width: e,
      height: t,
      widthSegments: i,
      heightSegments: s
    };
    const r = e / 2, o = t / 2, a = Math.floor(i), l = Math.floor(s), c = a + 1, u = l + 1, h = e / a, f = t / l, p = [], v = [], x = [], m = [];
    for (let d = 0; d < u; d++) {
      const b = d * f - o;
      for (let A = 0; A < c; A++) {
        const M = A * h - r;
        v.push(M, -b, 0), x.push(0, 0, 1), m.push(A / a), m.push(1 - d / l);
      }
    }
    for (let d = 0; d < l; d++)
      for (let b = 0; b < a; b++) {
        const A = b + c * d, M = b + c * (d + 1), C = b + 1 + c * (d + 1), w = b + 1 + c * d;
        p.push(A, M, w), p.push(M, C, w);
      }
    this.setIndex(p), this.setAttribute("position", new mt(v, 3)), this.setAttribute("normal", new mt(x, 3)), this.setAttribute("uv", new mt(m, 2));
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {PlaneGeometry} A new instance.
   */
  static fromJSON(e) {
    return new Hs(e.width, e.height, e.widthSegments, e.heightSegments);
  }
}
class Es extends Nt {
  /**
   * Constructs a new sphere geometry.
   *
   * @param {number} [radius=1] - The sphere radius.
   * @param {number} [widthSegments=32] - The number of horizontal segments. Minimum value is `3`.
   * @param {number} [heightSegments=16] - The number of vertical segments. Minimum value is `2`.
   * @param {number} [phiStart=0] - The horizontal starting angle in radians.
   * @param {number} [phiLength=Math.PI*2] - The horizontal sweep angle size.
   * @param {number} [thetaStart=0] - The vertical starting angle in radians.
   * @param {number} [thetaLength=Math.PI] - The vertical sweep angle size.
   */
  constructor(e = 1, t = 32, i = 16, s = 0, r = Math.PI * 2, o = 0, a = Math.PI) {
    super(), this.type = "SphereGeometry", this.parameters = {
      radius: e,
      widthSegments: t,
      heightSegments: i,
      phiStart: s,
      phiLength: r,
      thetaStart: o,
      thetaLength: a
    }, t = Math.max(3, Math.floor(t)), i = Math.max(2, Math.floor(i));
    const l = Math.min(o + a, Math.PI);
    let c = 0;
    const u = [], h = new N(), f = new N(), p = [], v = [], x = [], m = [];
    for (let d = 0; d <= i; d++) {
      const b = [], A = d / i;
      let M = 0;
      d === 0 && o === 0 ? M = 0.5 / t : d === i && l === Math.PI && (M = -0.5 / t);
      for (let C = 0; C <= t; C++) {
        const w = C / t;
        h.x = -e * Math.cos(s + w * r) * Math.sin(o + A * a), h.y = e * Math.cos(o + A * a), h.z = e * Math.sin(s + w * r) * Math.sin(o + A * a), v.push(h.x, h.y, h.z), f.copy(h).normalize(), x.push(f.x, f.y, f.z), m.push(w + M, 1 - A), b.push(c++);
      }
      u.push(b);
    }
    for (let d = 0; d < i; d++)
      for (let b = 0; b < t; b++) {
        const A = u[d][b + 1], M = u[d][b], C = u[d + 1][b], w = u[d + 1][b + 1];
        (d !== 0 || o > 0) && p.push(A, M, w), (d !== i - 1 || l < Math.PI) && p.push(M, C, w);
      }
    this.setIndex(p), this.setAttribute("position", new mt(v, 3)), this.setAttribute("normal", new mt(x, 3)), this.setAttribute("uv", new mt(m, 2));
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {SphereGeometry} A new instance.
   */
  static fromJSON(e) {
    return new Es(e.radius, e.widthSegments, e.heightSegments, e.phiStart, e.phiLength, e.thetaStart, e.thetaLength);
  }
}
class Ts extends Nt {
  /**
   * Constructs a new torus geometry.
   *
   * @param {number} [radius=1] - Radius of the torus, from the center of the torus to the center of the tube.
   * @param {number} [tube=0.4] - Radius of the tube. Must be smaller than `radius`.
   * @param {number} [radialSegments=12] - The number of radial segments.
   * @param {number} [tubularSegments=48] - The number of tubular segments.
   * @param {number} [arc=Math.PI*2] - Central angle in radians.
   */
  constructor(e = 1, t = 0.4, i = 12, s = 48, r = Math.PI * 2) {
    super(), this.type = "TorusGeometry", this.parameters = {
      radius: e,
      tube: t,
      radialSegments: i,
      tubularSegments: s,
      arc: r
    }, i = Math.floor(i), s = Math.floor(s);
    const o = [], a = [], l = [], c = [], u = new N(), h = new N(), f = new N();
    for (let p = 0; p <= i; p++)
      for (let v = 0; v <= s; v++) {
        const x = v / s * r, m = p / i * Math.PI * 2;
        h.x = (e + t * Math.cos(m)) * Math.cos(x), h.y = (e + t * Math.cos(m)) * Math.sin(x), h.z = t * Math.sin(m), a.push(h.x, h.y, h.z), u.x = e * Math.cos(x), u.y = e * Math.sin(x), f.subVectors(h, u).normalize(), l.push(f.x, f.y, f.z), c.push(v / s), c.push(p / i);
      }
    for (let p = 1; p <= i; p++)
      for (let v = 1; v <= s; v++) {
        const x = (s + 1) * p + v - 1, m = (s + 1) * (p - 1) + v - 1, d = (s + 1) * (p - 1) + v, b = (s + 1) * p + v;
        o.push(x, m, b), o.push(m, d, b);
      }
    this.setIndex(o), this.setAttribute("position", new mt(a, 3)), this.setAttribute("normal", new mt(l, 3)), this.setAttribute("uv", new mt(c, 2));
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {TorusGeometry} A new instance.
   */
  static fromJSON(e) {
    return new Ts(e.radius, e.tube, e.radialSegments, e.tubularSegments, e.arc);
  }
}
class bo extends Qi {
  /**
   * Constructs a new mesh standard material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isMeshStandardMaterial = !0, this.type = "MeshStandardMaterial", this.defines = { STANDARD: "" }, this.color = new Xe(16777215), this.roughness = 1, this.metalness = 0, this.map = null, this.lightMap = null, this.lightMapIntensity = 1, this.aoMap = null, this.aoMapIntensity = 1, this.emissive = new Xe(0), this.emissiveIntensity = 1, this.emissiveMap = null, this.bumpMap = null, this.bumpScale = 1, this.normalMap = null, this.normalMapType = fd, this.normalScale = new Ve(1, 1), this.displacementMap = null, this.displacementScale = 1, this.displacementBias = 0, this.roughnessMap = null, this.metalnessMap = null, this.alphaMap = null, this.envMap = null, this.envMapRotation = new zn(), this.envMapIntensity = 1, this.wireframe = !1, this.wireframeLinewidth = 1, this.wireframeLinecap = "round", this.wireframeLinejoin = "round", this.flatShading = !1, this.fog = !0, this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.defines = { STANDARD: "" }, this.color.copy(e.color), this.roughness = e.roughness, this.metalness = e.metalness, this.map = e.map, this.lightMap = e.lightMap, this.lightMapIntensity = e.lightMapIntensity, this.aoMap = e.aoMap, this.aoMapIntensity = e.aoMapIntensity, this.emissive.copy(e.emissive), this.emissiveMap = e.emissiveMap, this.emissiveIntensity = e.emissiveIntensity, this.bumpMap = e.bumpMap, this.bumpScale = e.bumpScale, this.normalMap = e.normalMap, this.normalMapType = e.normalMapType, this.normalScale.copy(e.normalScale), this.displacementMap = e.displacementMap, this.displacementScale = e.displacementScale, this.displacementBias = e.displacementBias, this.roughnessMap = e.roughnessMap, this.metalnessMap = e.metalnessMap, this.alphaMap = e.alphaMap, this.envMap = e.envMap, this.envMapRotation.copy(e.envMapRotation), this.envMapIntensity = e.envMapIntensity, this.wireframe = e.wireframe, this.wireframeLinewidth = e.wireframeLinewidth, this.wireframeLinecap = e.wireframeLinecap, this.wireframeLinejoin = e.wireframeLinejoin, this.flatShading = e.flatShading, this.fog = e.fog, this;
  }
}
class nh extends bo {
  /**
   * Constructs a new mesh physical material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isMeshPhysicalMaterial = !0, this.defines = {
      STANDARD: "",
      PHYSICAL: ""
    }, this.type = "MeshPhysicalMaterial", this.anisotropyRotation = 0, this.anisotropyMap = null, this.clearcoatMap = null, this.clearcoatRoughness = 0, this.clearcoatRoughnessMap = null, this.clearcoatNormalScale = new Ve(1, 1), this.clearcoatNormalMap = null, this.ior = 1.5, Object.defineProperty(this, "reflectivity", {
      get: function() {
        return Ke(2.5 * (this.ior - 1) / (this.ior + 1), 0, 1);
      },
      set: function(t) {
        this.ior = (1 + 0.4 * t) / (1 - 0.4 * t);
      }
    }), this.iridescenceMap = null, this.iridescenceIOR = 1.3, this.iridescenceThicknessRange = [100, 400], this.iridescenceThicknessMap = null, this.sheenColor = new Xe(0), this.sheenColorMap = null, this.sheenRoughness = 1, this.sheenRoughnessMap = null, this.transmissionMap = null, this.thickness = 0, this.thicknessMap = null, this.attenuationDistance = 1 / 0, this.attenuationColor = new Xe(1, 1, 1), this.specularIntensity = 1, this.specularIntensityMap = null, this.specularColor = new Xe(1, 1, 1), this.specularColorMap = null, this._anisotropy = 0, this._clearcoat = 0, this._dispersion = 0, this._iridescence = 0, this._sheen = 0, this._transmission = 0, this.setValues(e);
  }
  /**
   * The anisotropy strength.
   *
   * @type {number}
   * @default 0
   */
  get anisotropy() {
    return this._anisotropy;
  }
  set anisotropy(e) {
    this._anisotropy > 0 != e > 0 && this.version++, this._anisotropy = e;
  }
  /**
   * Represents the intensity of the clear coat layer, from `0.0` to `1.0`. Use
   * clear coat related properties to enable multilayer materials that have a
   * thin translucent layer over the base layer.
   *
   * @type {number}
   * @default 0
   */
  get clearcoat() {
    return this._clearcoat;
  }
  set clearcoat(e) {
    this._clearcoat > 0 != e > 0 && this.version++, this._clearcoat = e;
  }
  /**
   * The intensity of the iridescence layer, simulating RGB color shift based on the angle between
   * the surface and the viewer, from `0.0` to `1.0`.
   *
   * @type {number}
   * @default 0
   */
  get iridescence() {
    return this._iridescence;
  }
  set iridescence(e) {
    this._iridescence > 0 != e > 0 && this.version++, this._iridescence = e;
  }
  /**
   * Defines the strength of the angular separation of colors (chromatic aberration) transmitting
   * through a relatively clear volume. Any value zero or larger is valid, the typical range of
   * realistic values is `[0, 1]`. This property can be only be used with transmissive objects.
   *
   * @type {number}
   * @default 0
   */
  get dispersion() {
    return this._dispersion;
  }
  set dispersion(e) {
    this._dispersion > 0 != e > 0 && this.version++, this._dispersion = e;
  }
  /**
   * The intensity of the sheen layer, from `0.0` to `1.0`.
   *
   * @type {number}
   * @default 0
   */
  get sheen() {
    return this._sheen;
  }
  set sheen(e) {
    this._sheen > 0 != e > 0 && this.version++, this._sheen = e;
  }
  /**
   * Degree of transmission (or optical transparency), from `0.0` to `1.0`.
   *
   * Thin, transparent or semitransparent, plastic or glass materials remain
   * largely reflective even if they are fully transmissive. The transmission
   * property can be used to model these materials.
   *
   * When transmission is non-zero, `opacity` should be  set to `1`.
   *
   * @type {number}
   * @default 0
   */
  get transmission() {
    return this._transmission;
  }
  set transmission(e) {
    this._transmission > 0 != e > 0 && this.version++, this._transmission = e;
  }
  copy(e) {
    return super.copy(e), this.defines = {
      STANDARD: "",
      PHYSICAL: ""
    }, this.anisotropy = e.anisotropy, this.anisotropyRotation = e.anisotropyRotation, this.anisotropyMap = e.anisotropyMap, this.clearcoat = e.clearcoat, this.clearcoatMap = e.clearcoatMap, this.clearcoatRoughness = e.clearcoatRoughness, this.clearcoatRoughnessMap = e.clearcoatRoughnessMap, this.clearcoatNormalMap = e.clearcoatNormalMap, this.clearcoatNormalScale.copy(e.clearcoatNormalScale), this.dispersion = e.dispersion, this.ior = e.ior, this.iridescence = e.iridescence, this.iridescenceMap = e.iridescenceMap, this.iridescenceIOR = e.iridescenceIOR, this.iridescenceThicknessRange = [...e.iridescenceThicknessRange], this.iridescenceThicknessMap = e.iridescenceThicknessMap, this.sheen = e.sheen, this.sheenColor.copy(e.sheenColor), this.sheenColorMap = e.sheenColorMap, this.sheenRoughness = e.sheenRoughness, this.sheenRoughnessMap = e.sheenRoughnessMap, this.transmission = e.transmission, this.transmissionMap = e.transmissionMap, this.thickness = e.thickness, this.thicknessMap = e.thicknessMap, this.attenuationDistance = e.attenuationDistance, this.attenuationColor.copy(e.attenuationColor), this.specularIntensity = e.specularIntensity, this.specularIntensityMap = e.specularIntensityMap, this.specularColor.copy(e.specularColor), this.specularColorMap = e.specularColorMap, this;
  }
}
class f0 extends Qi {
  /**
   * Constructs a new mesh depth material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isMeshDepthMaterial = !0, this.type = "MeshDepthMaterial", this.depthPacking = Sg, this.map = null, this.alphaMap = null, this.displacementMap = null, this.displacementScale = 1, this.displacementBias = 0, this.wireframe = !1, this.wireframeLinewidth = 1, this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.depthPacking = e.depthPacking, this.map = e.map, this.alphaMap = e.alphaMap, this.displacementMap = e.displacementMap, this.displacementScale = e.displacementScale, this.displacementBias = e.displacementBias, this.wireframe = e.wireframe, this.wireframeLinewidth = e.wireframeLinewidth, this;
  }
}
class d0 extends Qi {
  /**
   * Constructs a new mesh distance material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isMeshDistanceMaterial = !0, this.type = "MeshDistanceMaterial", this.map = null, this.alphaMap = null, this.displacementMap = null, this.displacementScale = 1, this.displacementBias = 0, this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.map = e.map, this.alphaMap = e.alphaMap, this.displacementMap = e.displacementMap, this.displacementScale = e.displacementScale, this.displacementBias = e.displacementBias, this;
  }
}
class Lc extends Tt {
  /**
   * Constructs a new light.
   *
   * @param {(number|Color|string)} [color=0xffffff] - The light's color.
   * @param {number} [intensity=1] - The light's strength/intensity.
   */
  constructor(e, t = 1) {
    super(), this.isLight = !0, this.type = "Light", this.color = new Xe(e), this.intensity = t;
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   */
  dispose() {
  }
  copy(e, t) {
    return super.copy(e, t), this.color.copy(e.color), this.intensity = e.intensity, this;
  }
  toJSON(e) {
    const t = super.toJSON(e);
    return t.object.color = this.color.getHex(), t.object.intensity = this.intensity, this.groundColor !== void 0 && (t.object.groundColor = this.groundColor.getHex()), this.distance !== void 0 && (t.object.distance = this.distance), this.angle !== void 0 && (t.object.angle = this.angle), this.decay !== void 0 && (t.object.decay = this.decay), this.penumbra !== void 0 && (t.object.penumbra = this.penumbra), this.shadow !== void 0 && (t.object.shadow = this.shadow.toJSON()), this.target !== void 0 && (t.object.target = this.target.uuid), t;
  }
}
class p0 extends Lc {
  /**
   * Constructs a new hemisphere light.
   *
   * @param {(number|Color|string)} [skyColor=0xffffff] - The light's sky color.
   * @param {(number|Color|string)} [groundColor=0xffffff] - The light's ground color.
   * @param {number} [intensity=1] - The light's strength/intensity.
   */
  constructor(e, t, i) {
    super(e, i), this.isHemisphereLight = !0, this.type = "HemisphereLight", this.position.copy(Tt.DEFAULT_UP), this.updateMatrix(), this.groundColor = new Xe(t);
  }
  copy(e, t) {
    return super.copy(e, t), this.groundColor.copy(e.groundColor), this;
  }
}
const Xa = /* @__PURE__ */ new pt(), ih = /* @__PURE__ */ new N(), sh = /* @__PURE__ */ new N();
class wd {
  /**
   * Constructs a new light shadow.
   *
   * @param {Camera} camera - The light's view of the world.
   */
  constructor(e) {
    this.camera = e, this.intensity = 1, this.bias = 0, this.normalBias = 0, this.radius = 1, this.blurSamples = 8, this.mapSize = new Ve(512, 512), this.mapType = Bn, this.map = null, this.mapPass = null, this.matrix = new pt(), this.autoUpdate = !0, this.needsUpdate = !1, this._frustum = new wc(), this._frameExtents = new Ve(1, 1), this._viewportCount = 1, this._viewports = [
      new lt(0, 0, 1, 1)
    ];
  }
  /**
   * Used internally by the renderer to get the number of viewports that need
   * to be rendered for this shadow.
   *
   * @return {number} The viewport count.
   */
  getViewportCount() {
    return this._viewportCount;
  }
  /**
   * Gets the shadow cameras frustum. Used internally by the renderer to cull objects.
   *
   * @return {Frustum} The shadow camera frustum.
   */
  getFrustum() {
    return this._frustum;
  }
  /**
   * Update the matrices for the camera and shadow, used internally by the renderer.
   *
   * @param {Light} light - The light for which the shadow is being rendered.
   */
  updateMatrices(e) {
    const t = this.camera, i = this.matrix;
    ih.setFromMatrixPosition(e.matrixWorld), t.position.copy(ih), sh.setFromMatrixPosition(e.target.matrixWorld), t.lookAt(sh), t.updateMatrixWorld(), Xa.multiplyMatrices(t.projectionMatrix, t.matrixWorldInverse), this._frustum.setFromProjectionMatrix(Xa, t.coordinateSystem, t.reversedDepth), t.reversedDepth ? i.set(
      0.5,
      0,
      0,
      0.5,
      0,
      0.5,
      0,
      0.5,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ) : i.set(
      0.5,
      0,
      0,
      0.5,
      0,
      0.5,
      0,
      0.5,
      0,
      0,
      0.5,
      0.5,
      0,
      0,
      0,
      1
    ), i.multiply(Xa);
  }
  /**
   * Returns a viewport definition for the given viewport index.
   *
   * @param {number} viewportIndex - The viewport index.
   * @return {Vector4} The viewport.
   */
  getViewport(e) {
    return this._viewports[e];
  }
  /**
   * Returns the frame extends.
   *
   * @return {Vector2} The frame extends.
   */
  getFrameExtents() {
    return this._frameExtents;
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   */
  dispose() {
    this.map && this.map.dispose(), this.mapPass && this.mapPass.dispose();
  }
  /**
   * Copies the values of the given light shadow instance to this instance.
   *
   * @param {LightShadow} source - The light shadow to copy.
   * @return {LightShadow} A reference to this light shadow instance.
   */
  copy(e) {
    return this.camera = e.camera.clone(), this.intensity = e.intensity, this.bias = e.bias, this.radius = e.radius, this.autoUpdate = e.autoUpdate, this.needsUpdate = e.needsUpdate, this.normalBias = e.normalBias, this.blurSamples = e.blurSamples, this.mapSize.copy(e.mapSize), this;
  }
  /**
   * Returns a new light shadow instance with copied values from this instance.
   *
   * @return {LightShadow} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Serializes the light shadow into JSON.
   *
   * @return {Object} A JSON object representing the serialized light shadow.
   * @see {@link ObjectLoader#parse}
   */
  toJSON() {
    const e = {};
    return this.intensity !== 1 && (e.intensity = this.intensity), this.bias !== 0 && (e.bias = this.bias), this.normalBias !== 0 && (e.normalBias = this.normalBias), this.radius !== 1 && (e.radius = this.radius), (this.mapSize.x !== 512 || this.mapSize.y !== 512) && (e.mapSize = this.mapSize.toArray()), e.camera = this.camera.toJSON(!1).object, delete e.camera.matrix, e;
  }
}
const rh = /* @__PURE__ */ new pt(), tr = /* @__PURE__ */ new N(), Ya = /* @__PURE__ */ new N();
class m0 extends wd {
  /**
   * Constructs a new point light shadow.
   */
  constructor() {
    super(new rn(90, 1, 0.5, 500)), this.isPointLightShadow = !0, this._frameExtents = new Ve(4, 2), this._viewportCount = 6, this._viewports = [
      // These viewports map a cube-map onto a 2D texture with the
      // following orientation:
      //
      //  xzXZ
      //   y Y
      //
      // X - Positive x direction
      // x - Negative x direction
      // Y - Positive y direction
      // y - Negative y direction
      // Z - Positive z direction
      // z - Negative z direction
      // positive X
      new lt(2, 1, 1, 1),
      // negative X
      new lt(0, 1, 1, 1),
      // positive Z
      new lt(3, 1, 1, 1),
      // negative Z
      new lt(1, 1, 1, 1),
      // positive Y
      new lt(3, 0, 1, 1),
      // negative Y
      new lt(1, 0, 1, 1)
    ], this._cubeDirections = [
      new N(1, 0, 0),
      new N(-1, 0, 0),
      new N(0, 0, 1),
      new N(0, 0, -1),
      new N(0, 1, 0),
      new N(0, -1, 0)
    ], this._cubeUps = [
      new N(0, 1, 0),
      new N(0, 1, 0),
      new N(0, 1, 0),
      new N(0, 1, 0),
      new N(0, 0, 1),
      new N(0, 0, -1)
    ];
  }
  /**
   * Update the matrices for the camera and shadow, used internally by the renderer.
   *
   * @param {Light} light - The light for which the shadow is being rendered.
   * @param {number} [viewportIndex=0] - The viewport index.
   */
  updateMatrices(e, t = 0) {
    const i = this.camera, s = this.matrix, r = e.distance || i.far;
    r !== i.far && (i.far = r, i.updateProjectionMatrix()), tr.setFromMatrixPosition(e.matrixWorld), i.position.copy(tr), Ya.copy(i.position), Ya.add(this._cubeDirections[t]), i.up.copy(this._cubeUps[t]), i.lookAt(Ya), i.updateMatrixWorld(), s.makeTranslation(-tr.x, -tr.y, -tr.z), rh.multiplyMatrices(i.projectionMatrix, i.matrixWorldInverse), this._frustum.setFromProjectionMatrix(rh, i.coordinateSystem, i.reversedDepth);
  }
}
class _0 extends Lc {
  /**
   * Constructs a new point light.
   *
   * @param {(number|Color|string)} [color=0xffffff] - The light's color.
   * @param {number} [intensity=1] - The light's strength/intensity measured in candela (cd).
   * @param {number} [distance=0] - Maximum range of the light. `0` means no limit.
   * @param {number} [decay=2] - The amount the light dims along the distance of the light.
   */
  constructor(e, t, i = 0, s = 2) {
    super(e, t), this.isPointLight = !0, this.type = "PointLight", this.distance = i, this.decay = s, this.shadow = new m0();
  }
  /**
   * The light's power. Power is the luminous power of the light measured in lumens (lm).
   * Changing the power will also change the light's intensity.
   *
   * @type {number}
   */
  get power() {
    return this.intensity * 4 * Math.PI;
  }
  set power(e) {
    this.intensity = e / (4 * Math.PI);
  }
  dispose() {
    this.shadow.dispose();
  }
  copy(e, t) {
    return super.copy(e, t), this.distance = e.distance, this.decay = e.decay, this.shadow = e.shadow.clone(), this;
  }
}
class Rd extends Sd {
  /**
   * Constructs a new orthographic camera.
   *
   * @param {number} [left=-1] - The left plane of the camera's frustum.
   * @param {number} [right=1] - The right plane of the camera's frustum.
   * @param {number} [top=1] - The top plane of the camera's frustum.
   * @param {number} [bottom=-1] - The bottom plane of the camera's frustum.
   * @param {number} [near=0.1] - The camera's near plane.
   * @param {number} [far=2000] - The camera's far plane.
   */
  constructor(e = -1, t = 1, i = 1, s = -1, r = 0.1, o = 2e3) {
    super(), this.isOrthographicCamera = !0, this.type = "OrthographicCamera", this.zoom = 1, this.view = null, this.left = e, this.right = t, this.top = i, this.bottom = s, this.near = r, this.far = o, this.updateProjectionMatrix();
  }
  copy(e, t) {
    return super.copy(e, t), this.left = e.left, this.right = e.right, this.top = e.top, this.bottom = e.bottom, this.near = e.near, this.far = e.far, this.zoom = e.zoom, this.view = e.view === null ? null : Object.assign({}, e.view), this;
  }
  /**
   * Sets an offset in a larger frustum. This is useful for multi-window or
   * multi-monitor/multi-machine setups.
   *
   * @param {number} fullWidth - The full width of multiview setup.
   * @param {number} fullHeight - The full height of multiview setup.
   * @param {number} x - The horizontal offset of the subcamera.
   * @param {number} y - The vertical offset of the subcamera.
   * @param {number} width - The width of subcamera.
   * @param {number} height - The height of subcamera.
   * @see {@link PerspectiveCamera#setViewOffset}
   */
  setViewOffset(e, t, i, s, r, o) {
    this.view === null && (this.view = {
      enabled: !0,
      fullWidth: 1,
      fullHeight: 1,
      offsetX: 0,
      offsetY: 0,
      width: 1,
      height: 1
    }), this.view.enabled = !0, this.view.fullWidth = e, this.view.fullHeight = t, this.view.offsetX = i, this.view.offsetY = s, this.view.width = r, this.view.height = o, this.updateProjectionMatrix();
  }
  /**
   * Removes the view offset from the projection matrix.
   */
  clearViewOffset() {
    this.view !== null && (this.view.enabled = !1), this.updateProjectionMatrix();
  }
  /**
   * Updates the camera's projection matrix. Must be called after any change of
   * camera properties.
   */
  updateProjectionMatrix() {
    const e = (this.right - this.left) / (2 * this.zoom), t = (this.top - this.bottom) / (2 * this.zoom), i = (this.right + this.left) / 2, s = (this.top + this.bottom) / 2;
    let r = i - e, o = i + e, a = s + t, l = s - t;
    if (this.view !== null && this.view.enabled) {
      const c = (this.right - this.left) / this.view.fullWidth / this.zoom, u = (this.top - this.bottom) / this.view.fullHeight / this.zoom;
      r += c * this.view.offsetX, o = r + c * this.view.width, a -= u * this.view.offsetY, l = a - u * this.view.height;
    }
    this.projectionMatrix.makeOrthographic(r, o, a, l, this.near, this.far, this.coordinateSystem, this.reversedDepth), this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
  }
  toJSON(e) {
    const t = super.toJSON(e);
    return t.object.zoom = this.zoom, t.object.left = this.left, t.object.right = this.right, t.object.top = this.top, t.object.bottom = this.bottom, t.object.near = this.near, t.object.far = this.far, this.view !== null && (t.object.view = Object.assign({}, this.view)), t;
  }
}
class g0 extends wd {
  /**
   * Constructs a new directional light shadow.
   */
  constructor() {
    super(new Rd(-5, 5, 5, -5, 0.5, 500)), this.isDirectionalLightShadow = !0;
  }
}
class v0 extends Lc {
  /**
   * Constructs a new directional light.
   *
   * @param {(number|Color|string)} [color=0xffffff] - The light's color.
   * @param {number} [intensity=1] - The light's strength/intensity.
   */
  constructor(e, t) {
    super(e, t), this.isDirectionalLight = !0, this.type = "DirectionalLight", this.position.copy(Tt.DEFAULT_UP), this.updateMatrix(), this.target = new Tt(), this.shadow = new g0();
  }
  dispose() {
    this.shadow.dispose();
  }
  copy(e) {
    return super.copy(e), this.target = e.target.clone(), this.shadow = e.shadow.clone(), this;
  }
}
class x0 extends rn {
  /**
   * Constructs a new array camera.
   *
   * @param {Array<PerspectiveCamera>} [array=[]] - An array of perspective sub cameras.
   */
  constructor(e = []) {
    super(), this.isArrayCamera = !0, this.isMultiViewCamera = !1, this.cameras = e;
  }
}
class M0 {
  /**
   * Constructs a new clock.
   *
   * @param {boolean} [autoStart=true] - Whether to automatically start the clock when
   * `getDelta()` is called for the first time.
   */
  constructor(e = !0) {
    this.autoStart = e, this.startTime = 0, this.oldTime = 0, this.elapsedTime = 0, this.running = !1;
  }
  /**
   * Starts the clock. When `autoStart` is set to `true`, the method is automatically
   * called by the class.
   */
  start() {
    this.startTime = performance.now(), this.oldTime = this.startTime, this.elapsedTime = 0, this.running = !0;
  }
  /**
   * Stops the clock.
   */
  stop() {
    this.getElapsedTime(), this.running = !1, this.autoStart = !1;
  }
  /**
   * Returns the elapsed time in seconds.
   *
   * @return {number} The elapsed time.
   */
  getElapsedTime() {
    return this.getDelta(), this.elapsedTime;
  }
  /**
   * Returns the delta time in seconds.
   *
   * @return {number} The delta time.
   */
  getDelta() {
    let e = 0;
    if (this.autoStart && !this.running)
      return this.start(), 0;
    if (this.running) {
      const t = performance.now();
      e = (t - this.oldTime) / 1e3, this.oldTime = t, this.elapsedTime += e;
    }
    return e;
  }
}
class oh {
  /**
   * Constructs a new spherical.
   *
   * @param {number} [radius=1] - The radius, or the Euclidean distance (straight-line distance) from the point to the origin.
   * @param {number} [phi=0] - The polar angle in radians from the y (up) axis.
   * @param {number} [theta=0] - The equator/azimuthal angle in radians around the y (up) axis.
   */
  constructor(e = 1, t = 0, i = 0) {
    this.radius = e, this.phi = t, this.theta = i;
  }
  /**
   * Sets the spherical components by copying the given values.
   *
   * @param {number} radius - The radius.
   * @param {number} phi - The polar angle.
   * @param {number} theta - The azimuthal angle.
   * @return {Spherical} A reference to this spherical.
   */
  set(e, t, i) {
    return this.radius = e, this.phi = t, this.theta = i, this;
  }
  /**
   * Copies the values of the given spherical to this instance.
   *
   * @param {Spherical} other - The spherical to copy.
   * @return {Spherical} A reference to this spherical.
   */
  copy(e) {
    return this.radius = e.radius, this.phi = e.phi, this.theta = e.theta, this;
  }
  /**
   * Restricts the polar angle [page:.phi phi] to be between `0.000001` and pi -
   * `0.000001`.
   *
   * @return {Spherical} A reference to this spherical.
   */
  makeSafe() {
    return this.phi = Ke(this.phi, 1e-6, Math.PI - 1e-6), this;
  }
  /**
   * Sets the spherical components from the given vector which is assumed to hold
   * Cartesian coordinates.
   *
   * @param {Vector3} v - The vector to set.
   * @return {Spherical} A reference to this spherical.
   */
  setFromVector3(e) {
    return this.setFromCartesianCoords(e.x, e.y, e.z);
  }
  /**
   * Sets the spherical components from the given Cartesian coordinates.
   *
   * @param {number} x - The x value.
   * @param {number} y - The y value.
   * @param {number} z - The z value.
   * @return {Spherical} A reference to this spherical.
   */
  setFromCartesianCoords(e, t, i) {
    return this.radius = Math.sqrt(e * e + t * t + i * i), this.radius === 0 ? (this.theta = 0, this.phi = 0) : (this.theta = Math.atan2(e, i), this.phi = Math.acos(Ke(t / this.radius, -1, 1))), this;
  }
  /**
   * Returns a new spherical with copied values from this instance.
   *
   * @return {Spherical} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
}
class ah extends Ed {
  /**
   * Constructs a new grid helper.
   *
   * @param {number} [size=10] - The size of the grid.
   * @param {number} [divisions=10] - The number of divisions across the grid.
   * @param {number|Color|string} [color1=0x444444] - The color of the center line.
   * @param {number|Color|string} [color2=0x888888] - The color of the lines of the grid.
   */
  constructor(e = 10, t = 10, i = 4473924, s = 8947848) {
    i = new Xe(i), s = new Xe(s);
    const r = t / 2, o = e / t, a = e / 2, l = [], c = [];
    for (let f = 0, p = 0, v = -a; f <= t; f++, v += o) {
      l.push(-a, 0, v, a, 0, v), l.push(v, 0, -a, v, 0, a);
      const x = f === r ? i : s;
      x.toArray(c, p), p += 3, x.toArray(c, p), p += 3, x.toArray(c, p), p += 3, x.toArray(c, p), p += 3;
    }
    const u = new Nt();
    u.setAttribute("position", new mt(l, 3)), u.setAttribute("color", new mt(c, 3));
    const h = new Rc({ vertexColors: !0, toneMapped: !1 });
    super(u, h), this.type = "GridHelper";
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   */
  dispose() {
    this.geometry.dispose(), this.material.dispose();
  }
}
class S0 extends Ji {
  /**
   * Constructs a new controls instance.
   *
   * @param {Object3D} object - The object that is managed by the controls.
   * @param {?HTMLDOMElement} domElement - The HTML element used for event listeners.
   */
  constructor(e, t = null) {
    super(), this.object = e, this.domElement = t, this.enabled = !0, this.state = -1, this.keys = {}, this.mouseButtons = { LEFT: null, MIDDLE: null, RIGHT: null }, this.touches = { ONE: null, TWO: null };
  }
  /**
   * Connects the controls to the DOM. This method has so called "side effects" since
   * it adds the module's event listeners to the DOM.
   *
   * @param {HTMLDOMElement} element - The DOM element to connect to.
   */
  connect(e) {
    if (e === void 0) {
      console.warn("THREE.Controls: connect() now requires an element.");
      return;
    }
    this.domElement !== null && this.disconnect(), this.domElement = e;
  }
  /**
   * Disconnects the controls from the DOM.
   */
  disconnect() {
  }
  /**
   * Call this method if you no longer want use to the controls. It frees all internal
   * resources and removes all event listeners.
   */
  dispose() {
  }
  /**
   * Controls should implement this method if they have to update their internal state
   * per simulation step.
   *
   * @param {number} [delta] - The time delta in seconds.
   */
  update() {
  }
}
function lh(n, e, t, i) {
  const s = y0(i);
  switch (t) {
    // https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glTexImage2D.xhtml
    case ld:
      return n * e;
    case ud:
      return n * e / s.components * s.byteLength;
    case Ec:
      return n * e / s.components * s.byteLength;
    case hd:
      return n * e * 2 / s.components * s.byteLength;
    case Tc:
      return n * e * 2 / s.components * s.byteLength;
    case cd:
      return n * e * 3 / s.components * s.byteLength;
    case xn:
      return n * e * 4 / s.components * s.byteLength;
    case bc:
      return n * e * 4 / s.components * s.byteLength;
    // https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_s3tc_srgb/
    case So:
    case yo:
      return Math.floor((n + 3) / 4) * Math.floor((e + 3) / 4) * 8;
    case Eo:
    case To:
      return Math.floor((n + 3) / 4) * Math.floor((e + 3) / 4) * 16;
    // https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_pvrtc/
    case Al:
    case Rl:
      return Math.max(n, 16) * Math.max(e, 8) / 4;
    case bl:
    case wl:
      return Math.max(n, 8) * Math.max(e, 8) / 2;
    // https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_etc/
    case Cl:
    case Pl:
      return Math.floor((n + 3) / 4) * Math.floor((e + 3) / 4) * 8;
    case Dl:
      return Math.floor((n + 3) / 4) * Math.floor((e + 3) / 4) * 16;
    // https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_astc/
    case Ll:
      return Math.floor((n + 3) / 4) * Math.floor((e + 3) / 4) * 16;
    case Il:
      return Math.floor((n + 4) / 5) * Math.floor((e + 3) / 4) * 16;
    case Ul:
      return Math.floor((n + 4) / 5) * Math.floor((e + 4) / 5) * 16;
    case Nl:
      return Math.floor((n + 5) / 6) * Math.floor((e + 4) / 5) * 16;
    case Fl:
      return Math.floor((n + 5) / 6) * Math.floor((e + 5) / 6) * 16;
    case Ol:
      return Math.floor((n + 7) / 8) * Math.floor((e + 4) / 5) * 16;
    case Bl:
      return Math.floor((n + 7) / 8) * Math.floor((e + 5) / 6) * 16;
    case zl:
      return Math.floor((n + 7) / 8) * Math.floor((e + 7) / 8) * 16;
    case Hl:
      return Math.floor((n + 9) / 10) * Math.floor((e + 4) / 5) * 16;
    case Vl:
      return Math.floor((n + 9) / 10) * Math.floor((e + 5) / 6) * 16;
    case kl:
      return Math.floor((n + 9) / 10) * Math.floor((e + 7) / 8) * 16;
    case Gl:
      return Math.floor((n + 9) / 10) * Math.floor((e + 9) / 10) * 16;
    case Wl:
      return Math.floor((n + 11) / 12) * Math.floor((e + 9) / 10) * 16;
    case Xl:
      return Math.floor((n + 11) / 12) * Math.floor((e + 11) / 12) * 16;
    // https://registry.khronos.org/webgl/extensions/EXT_texture_compression_bptc/
    case Yl:
    case ql:
    case jl:
      return Math.ceil(n / 4) * Math.ceil(e / 4) * 16;
    // https://registry.khronos.org/webgl/extensions/EXT_texture_compression_rgtc/
    case Kl:
    case $l:
      return Math.ceil(n / 4) * Math.ceil(e / 4) * 8;
    case Zl:
    case Jl:
      return Math.ceil(n / 4) * Math.ceil(e / 4) * 16;
  }
  throw new Error(
    `Unable to determine texture byte length for ${t} format.`
  );
}
function y0(n) {
  switch (n) {
    case Bn:
    case sd:
      return { byteLength: 1, components: 1 };
    case br:
    case rd:
    case Ir:
      return { byteLength: 2, components: 1 };
    case Sc:
    case yc:
      return { byteLength: 2, components: 4 };
    case Yi:
    case Mc:
    case ei:
      return { byteLength: 4, components: 1 };
    case od:
    case ad:
      return { byteLength: 4, components: 3 };
  }
  throw new Error(`Unknown texture type ${n}.`);
}
typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register", { detail: {
  revision: xc
} }));
typeof window < "u" && (window.__THREE__ ? console.warn("WARNING: Multiple instances of Three.js being imported.") : window.__THREE__ = xc);
function Cd() {
  let n = null, e = !1, t = null, i = null;
  function s(r, o) {
    t(r, o), i = n.requestAnimationFrame(s);
  }
  return {
    start: function() {
      e !== !0 && t !== null && (i = n.requestAnimationFrame(s), e = !0);
    },
    stop: function() {
      n.cancelAnimationFrame(i), e = !1;
    },
    setAnimationLoop: function(r) {
      t = r;
    },
    setContext: function(r) {
      n = r;
    }
  };
}
function E0(n) {
  const e = /* @__PURE__ */ new WeakMap();
  function t(a, l) {
    const c = a.array, u = a.usage, h = c.byteLength, f = n.createBuffer();
    n.bindBuffer(l, f), n.bufferData(l, c, u), a.onUploadCallback();
    let p;
    if (c instanceof Float32Array)
      p = n.FLOAT;
    else if (typeof Float16Array < "u" && c instanceof Float16Array)
      p = n.HALF_FLOAT;
    else if (c instanceof Uint16Array)
      a.isFloat16BufferAttribute ? p = n.HALF_FLOAT : p = n.UNSIGNED_SHORT;
    else if (c instanceof Int16Array)
      p = n.SHORT;
    else if (c instanceof Uint32Array)
      p = n.UNSIGNED_INT;
    else if (c instanceof Int32Array)
      p = n.INT;
    else if (c instanceof Int8Array)
      p = n.BYTE;
    else if (c instanceof Uint8Array)
      p = n.UNSIGNED_BYTE;
    else if (c instanceof Uint8ClampedArray)
      p = n.UNSIGNED_BYTE;
    else
      throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: " + c);
    return {
      buffer: f,
      type: p,
      bytesPerElement: c.BYTES_PER_ELEMENT,
      version: a.version,
      size: h
    };
  }
  function i(a, l, c) {
    const u = l.array, h = l.updateRanges;
    if (n.bindBuffer(c, a), h.length === 0)
      n.bufferSubData(c, 0, u);
    else {
      h.sort((p, v) => p.start - v.start);
      let f = 0;
      for (let p = 1; p < h.length; p++) {
        const v = h[f], x = h[p];
        x.start <= v.start + v.count + 1 ? v.count = Math.max(
          v.count,
          x.start + x.count - v.start
        ) : (++f, h[f] = x);
      }
      h.length = f + 1;
      for (let p = 0, v = h.length; p < v; p++) {
        const x = h[p];
        n.bufferSubData(
          c,
          x.start * u.BYTES_PER_ELEMENT,
          u,
          x.start,
          x.count
        );
      }
      l.clearUpdateRanges();
    }
    l.onUploadCallback();
  }
  function s(a) {
    return a.isInterleavedBufferAttribute && (a = a.data), e.get(a);
  }
  function r(a) {
    a.isInterleavedBufferAttribute && (a = a.data);
    const l = e.get(a);
    l && (n.deleteBuffer(l.buffer), e.delete(a));
  }
  function o(a, l) {
    if (a.isInterleavedBufferAttribute && (a = a.data), a.isGLBufferAttribute) {
      const u = e.get(a);
      (!u || u.version < a.version) && e.set(a, {
        buffer: a.buffer,
        type: a.type,
        bytesPerElement: a.elementSize,
        version: a.version
      });
      return;
    }
    const c = e.get(a);
    if (c === void 0)
      e.set(a, t(a, l));
    else if (c.version < a.version) {
      if (c.size !== a.array.byteLength)
        throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");
      i(c.buffer, a, l), c.version = a.version;
    }
  }
  return {
    get: s,
    remove: r,
    update: o
  };
}
var T0 = `#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`, b0 = `#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`, A0 = `#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`, w0 = `#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`, R0 = `#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`, C0 = `#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`, P0 = `#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT )
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN )
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`, D0 = `#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`, L0 = `#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec3 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 ).rgb;
	}
#endif`, I0 = `#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`, U0 = `vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`, N0 = `vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`, F0 = `float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`, O0 = `#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`, B0 = `#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`, z0 = `#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`, H0 = `#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`, V0 = `#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`, k0 = `#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`, G0 = `#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`, W0 = `#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`, X0 = `#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`, Y0 = `#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
	vec3 batchingColor = getBatchingColor( getIndirectIndex( gl_DrawID ) );
	vColor.xyz *= batchingColor.xyz;
#endif`, q0 = `#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`, j0 = `#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`, K0 = `vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`, $0 = `#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`, Z0 = `#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`, J0 = `#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`, Q0 = `#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`, ev = "gl_FragColor = linearToOutputTexel( gl_FragColor );", tv = `vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`, nv = `#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`, iv = `#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif

#endif`, sv = `#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`, rv = `#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS

		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`, ov = `#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`, av = `#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`, lv = `#ifdef USE_FOG
	varying float vFogDepth;
#endif`, cv = `#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`, uv = `#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`, hv = `#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`, fv = `#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`, dv = `LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`, pv = `varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`, mv = `uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`, _v = `#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, roughness * roughness) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`, gv = `ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`, vv = `varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`, xv = `BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`, Mv = `varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`, Sv = `PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`, yv = `struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return saturate(v);
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );
	vec4 r = roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;
	vec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;
	return fab;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`, Ev = `
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`, Tv = `#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`, bv = `#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`, Av = `#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`, wv = `#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`, Rv = `#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`, Cv = `#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`, Pv = `#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`, Dv = `#ifdef USE_MAP
	uniform sampler2D map;
#endif`, Lv = `#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`, Iv = `#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`, Uv = `float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`, Nv = `#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`, Fv = `#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`, Ov = `#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`, Bv = `#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`, zv = `#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`, Hv = `#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`, Vv = `float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`, kv = `#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`, Gv = `#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`, Wv = `#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`, Xv = `#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`, Yv = `#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`, qv = `#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`, jv = `#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`, Kv = `#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`, $v = `#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`, Zv = `#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`, Jv = `vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`, Qv = `#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`, ex = `vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`, tx = `#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`, nx = `#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`, ix = `float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`, sx = `#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`, rx = `#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		float depth = unpackRGBAToDepth( texture2D( depths, uv ) );
		#ifdef USE_REVERSED_DEPTH_BUFFER
			return step( depth, compare );
		#else
			return step( compare, depth );
		#endif
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow( sampler2D shadow, vec2 uv, float compare ) {
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		#ifdef USE_REVERSED_DEPTH_BUFFER
			float hard_shadow = step( distribution.x, compare );
		#else
			float hard_shadow = step( compare, distribution.x );
		#endif
		if ( hard_shadow != 1.0 ) {
			float distance = compare - distribution.x;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;

		float lightToPositionLength = length( lightToPosition );
		if ( lightToPositionLength - shadowCameraFar <= 0.0 && lightToPositionLength - shadowCameraNear >= 0.0 ) {
			float dp = ( lightToPositionLength - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
			#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
				vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
				shadow = (
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
				) * ( 1.0 / 9.0 );
			#else
				shadow = texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
			#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
#endif`, ox = `#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`, ax = `#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`, lx = `float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`, cx = `#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`, ux = `#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`, hx = `#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`, fx = `#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`, dx = `float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`, px = `#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`, mx = `#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`, _x = `#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`, gx = `#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`, vx = `#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`, xx = `#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`, Mx = `#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`, Sx = `#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`, yx = `#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;
const Ex = `varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`, Tx = `uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`, bx = `varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`, Ax = `#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`, wx = `varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`, Rx = `uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`, Cx = `#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`, Px = `#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`, Dx = `#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`, Lx = `#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`, Ix = `varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`, Ux = `uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`, Nx = `uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`, Fx = `uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`, Ox = `#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`, Bx = `uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, zx = `#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`, Hx = `#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, Vx = `#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`, kx = `#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, Gx = `#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`, Wx = `#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`, Xx = `#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`, Yx = `#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, qx = `#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`, jx = `#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecularDirect + sheenSpecularIndirect;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, Kx = `#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`, $x = `#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, Zx = `uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`, Jx = `uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`, Qx = `#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`, eM = `uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`, tM = `uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`, nM = `uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`, je = {
  alphahash_fragment: T0,
  alphahash_pars_fragment: b0,
  alphamap_fragment: A0,
  alphamap_pars_fragment: w0,
  alphatest_fragment: R0,
  alphatest_pars_fragment: C0,
  aomap_fragment: P0,
  aomap_pars_fragment: D0,
  batching_pars_vertex: L0,
  batching_vertex: I0,
  begin_vertex: U0,
  beginnormal_vertex: N0,
  bsdfs: F0,
  iridescence_fragment: O0,
  bumpmap_pars_fragment: B0,
  clipping_planes_fragment: z0,
  clipping_planes_pars_fragment: H0,
  clipping_planes_pars_vertex: V0,
  clipping_planes_vertex: k0,
  color_fragment: G0,
  color_pars_fragment: W0,
  color_pars_vertex: X0,
  color_vertex: Y0,
  common: q0,
  cube_uv_reflection_fragment: j0,
  defaultnormal_vertex: K0,
  displacementmap_pars_vertex: $0,
  displacementmap_vertex: Z0,
  emissivemap_fragment: J0,
  emissivemap_pars_fragment: Q0,
  colorspace_fragment: ev,
  colorspace_pars_fragment: tv,
  envmap_fragment: nv,
  envmap_common_pars_fragment: iv,
  envmap_pars_fragment: sv,
  envmap_pars_vertex: rv,
  envmap_physical_pars_fragment: _v,
  envmap_vertex: ov,
  fog_vertex: av,
  fog_pars_vertex: lv,
  fog_fragment: cv,
  fog_pars_fragment: uv,
  gradientmap_pars_fragment: hv,
  lightmap_pars_fragment: fv,
  lights_lambert_fragment: dv,
  lights_lambert_pars_fragment: pv,
  lights_pars_begin: mv,
  lights_toon_fragment: gv,
  lights_toon_pars_fragment: vv,
  lights_phong_fragment: xv,
  lights_phong_pars_fragment: Mv,
  lights_physical_fragment: Sv,
  lights_physical_pars_fragment: yv,
  lights_fragment_begin: Ev,
  lights_fragment_maps: Tv,
  lights_fragment_end: bv,
  logdepthbuf_fragment: Av,
  logdepthbuf_pars_fragment: wv,
  logdepthbuf_pars_vertex: Rv,
  logdepthbuf_vertex: Cv,
  map_fragment: Pv,
  map_pars_fragment: Dv,
  map_particle_fragment: Lv,
  map_particle_pars_fragment: Iv,
  metalnessmap_fragment: Uv,
  metalnessmap_pars_fragment: Nv,
  morphinstance_vertex: Fv,
  morphcolor_vertex: Ov,
  morphnormal_vertex: Bv,
  morphtarget_pars_vertex: zv,
  morphtarget_vertex: Hv,
  normal_fragment_begin: Vv,
  normal_fragment_maps: kv,
  normal_pars_fragment: Gv,
  normal_pars_vertex: Wv,
  normal_vertex: Xv,
  normalmap_pars_fragment: Yv,
  clearcoat_normal_fragment_begin: qv,
  clearcoat_normal_fragment_maps: jv,
  clearcoat_pars_fragment: Kv,
  iridescence_pars_fragment: $v,
  opaque_fragment: Zv,
  packing: Jv,
  premultiplied_alpha_fragment: Qv,
  project_vertex: ex,
  dithering_fragment: tx,
  dithering_pars_fragment: nx,
  roughnessmap_fragment: ix,
  roughnessmap_pars_fragment: sx,
  shadowmap_pars_fragment: rx,
  shadowmap_pars_vertex: ox,
  shadowmap_vertex: ax,
  shadowmask_pars_fragment: lx,
  skinbase_vertex: cx,
  skinning_pars_vertex: ux,
  skinning_vertex: hx,
  skinnormal_vertex: fx,
  specularmap_fragment: dx,
  specularmap_pars_fragment: px,
  tonemapping_fragment: mx,
  tonemapping_pars_fragment: _x,
  transmission_fragment: gx,
  transmission_pars_fragment: vx,
  uv_pars_fragment: xx,
  uv_pars_vertex: Mx,
  uv_vertex: Sx,
  worldpos_vertex: yx,
  background_vert: Ex,
  background_frag: Tx,
  backgroundCube_vert: bx,
  backgroundCube_frag: Ax,
  cube_vert: wx,
  cube_frag: Rx,
  depth_vert: Cx,
  depth_frag: Px,
  distanceRGBA_vert: Dx,
  distanceRGBA_frag: Lx,
  equirect_vert: Ix,
  equirect_frag: Ux,
  linedashed_vert: Nx,
  linedashed_frag: Fx,
  meshbasic_vert: Ox,
  meshbasic_frag: Bx,
  meshlambert_vert: zx,
  meshlambert_frag: Hx,
  meshmatcap_vert: Vx,
  meshmatcap_frag: kx,
  meshnormal_vert: Gx,
  meshnormal_frag: Wx,
  meshphong_vert: Xx,
  meshphong_frag: Yx,
  meshphysical_vert: qx,
  meshphysical_frag: jx,
  meshtoon_vert: Kx,
  meshtoon_frag: $x,
  points_vert: Zx,
  points_frag: Jx,
  shadow_vert: Qx,
  shadow_frag: eM,
  sprite_vert: tM,
  sprite_frag: nM
}, _e = {
  common: {
    diffuse: { value: /* @__PURE__ */ new Xe(16777215) },
    opacity: { value: 1 },
    map: { value: null },
    mapTransform: { value: /* @__PURE__ */ new qe() },
    alphaMap: { value: null },
    alphaMapTransform: { value: /* @__PURE__ */ new qe() },
    alphaTest: { value: 0 }
  },
  specularmap: {
    specularMap: { value: null },
    specularMapTransform: { value: /* @__PURE__ */ new qe() }
  },
  envmap: {
    envMap: { value: null },
    envMapRotation: { value: /* @__PURE__ */ new qe() },
    flipEnvMap: { value: -1 },
    reflectivity: { value: 1 },
    // basic, lambert, phong
    ior: { value: 1.5 },
    // physical
    refractionRatio: { value: 0.98 }
    // basic, lambert, phong
  },
  aomap: {
    aoMap: { value: null },
    aoMapIntensity: { value: 1 },
    aoMapTransform: { value: /* @__PURE__ */ new qe() }
  },
  lightmap: {
    lightMap: { value: null },
    lightMapIntensity: { value: 1 },
    lightMapTransform: { value: /* @__PURE__ */ new qe() }
  },
  bumpmap: {
    bumpMap: { value: null },
    bumpMapTransform: { value: /* @__PURE__ */ new qe() },
    bumpScale: { value: 1 }
  },
  normalmap: {
    normalMap: { value: null },
    normalMapTransform: { value: /* @__PURE__ */ new qe() },
    normalScale: { value: /* @__PURE__ */ new Ve(1, 1) }
  },
  displacementmap: {
    displacementMap: { value: null },
    displacementMapTransform: { value: /* @__PURE__ */ new qe() },
    displacementScale: { value: 1 },
    displacementBias: { value: 0 }
  },
  emissivemap: {
    emissiveMap: { value: null },
    emissiveMapTransform: { value: /* @__PURE__ */ new qe() }
  },
  metalnessmap: {
    metalnessMap: { value: null },
    metalnessMapTransform: { value: /* @__PURE__ */ new qe() }
  },
  roughnessmap: {
    roughnessMap: { value: null },
    roughnessMapTransform: { value: /* @__PURE__ */ new qe() }
  },
  gradientmap: {
    gradientMap: { value: null }
  },
  fog: {
    fogDensity: { value: 25e-5 },
    fogNear: { value: 1 },
    fogFar: { value: 2e3 },
    fogColor: { value: /* @__PURE__ */ new Xe(16777215) }
  },
  lights: {
    ambientLightColor: { value: [] },
    lightProbe: { value: [] },
    directionalLights: { value: [], properties: {
      direction: {},
      color: {}
    } },
    directionalLightShadows: { value: [], properties: {
      shadowIntensity: 1,
      shadowBias: {},
      shadowNormalBias: {},
      shadowRadius: {},
      shadowMapSize: {}
    } },
    directionalShadowMap: { value: [] },
    directionalShadowMatrix: { value: [] },
    spotLights: { value: [], properties: {
      color: {},
      position: {},
      direction: {},
      distance: {},
      coneCos: {},
      penumbraCos: {},
      decay: {}
    } },
    spotLightShadows: { value: [], properties: {
      shadowIntensity: 1,
      shadowBias: {},
      shadowNormalBias: {},
      shadowRadius: {},
      shadowMapSize: {}
    } },
    spotLightMap: { value: [] },
    spotShadowMap: { value: [] },
    spotLightMatrix: { value: [] },
    pointLights: { value: [], properties: {
      color: {},
      position: {},
      decay: {},
      distance: {}
    } },
    pointLightShadows: { value: [], properties: {
      shadowIntensity: 1,
      shadowBias: {},
      shadowNormalBias: {},
      shadowRadius: {},
      shadowMapSize: {},
      shadowCameraNear: {},
      shadowCameraFar: {}
    } },
    pointShadowMap: { value: [] },
    pointShadowMatrix: { value: [] },
    hemisphereLights: { value: [], properties: {
      direction: {},
      skyColor: {},
      groundColor: {}
    } },
    // TODO (abelnation): RectAreaLight BRDF data needs to be moved from example to main src
    rectAreaLights: { value: [], properties: {
      color: {},
      position: {},
      width: {},
      height: {}
    } },
    ltc_1: { value: null },
    ltc_2: { value: null }
  },
  points: {
    diffuse: { value: /* @__PURE__ */ new Xe(16777215) },
    opacity: { value: 1 },
    size: { value: 1 },
    scale: { value: 1 },
    map: { value: null },
    alphaMap: { value: null },
    alphaMapTransform: { value: /* @__PURE__ */ new qe() },
    alphaTest: { value: 0 },
    uvTransform: { value: /* @__PURE__ */ new qe() }
  },
  sprite: {
    diffuse: { value: /* @__PURE__ */ new Xe(16777215) },
    opacity: { value: 1 },
    center: { value: /* @__PURE__ */ new Ve(0.5, 0.5) },
    rotation: { value: 0 },
    map: { value: null },
    mapTransform: { value: /* @__PURE__ */ new qe() },
    alphaMap: { value: null },
    alphaMapTransform: { value: /* @__PURE__ */ new qe() },
    alphaTest: { value: 0 }
  }
}, Ln = {
  basic: {
    uniforms: /* @__PURE__ */ zt([
      _e.common,
      _e.specularmap,
      _e.envmap,
      _e.aomap,
      _e.lightmap,
      _e.fog
    ]),
    vertexShader: je.meshbasic_vert,
    fragmentShader: je.meshbasic_frag
  },
  lambert: {
    uniforms: /* @__PURE__ */ zt([
      _e.common,
      _e.specularmap,
      _e.envmap,
      _e.aomap,
      _e.lightmap,
      _e.emissivemap,
      _e.bumpmap,
      _e.normalmap,
      _e.displacementmap,
      _e.fog,
      _e.lights,
      {
        emissive: { value: /* @__PURE__ */ new Xe(0) }
      }
    ]),
    vertexShader: je.meshlambert_vert,
    fragmentShader: je.meshlambert_frag
  },
  phong: {
    uniforms: /* @__PURE__ */ zt([
      _e.common,
      _e.specularmap,
      _e.envmap,
      _e.aomap,
      _e.lightmap,
      _e.emissivemap,
      _e.bumpmap,
      _e.normalmap,
      _e.displacementmap,
      _e.fog,
      _e.lights,
      {
        emissive: { value: /* @__PURE__ */ new Xe(0) },
        specular: { value: /* @__PURE__ */ new Xe(1118481) },
        shininess: { value: 30 }
      }
    ]),
    vertexShader: je.meshphong_vert,
    fragmentShader: je.meshphong_frag
  },
  standard: {
    uniforms: /* @__PURE__ */ zt([
      _e.common,
      _e.envmap,
      _e.aomap,
      _e.lightmap,
      _e.emissivemap,
      _e.bumpmap,
      _e.normalmap,
      _e.displacementmap,
      _e.roughnessmap,
      _e.metalnessmap,
      _e.fog,
      _e.lights,
      {
        emissive: { value: /* @__PURE__ */ new Xe(0) },
        roughness: { value: 1 },
        metalness: { value: 0 },
        envMapIntensity: { value: 1 }
      }
    ]),
    vertexShader: je.meshphysical_vert,
    fragmentShader: je.meshphysical_frag
  },
  toon: {
    uniforms: /* @__PURE__ */ zt([
      _e.common,
      _e.aomap,
      _e.lightmap,
      _e.emissivemap,
      _e.bumpmap,
      _e.normalmap,
      _e.displacementmap,
      _e.gradientmap,
      _e.fog,
      _e.lights,
      {
        emissive: { value: /* @__PURE__ */ new Xe(0) }
      }
    ]),
    vertexShader: je.meshtoon_vert,
    fragmentShader: je.meshtoon_frag
  },
  matcap: {
    uniforms: /* @__PURE__ */ zt([
      _e.common,
      _e.bumpmap,
      _e.normalmap,
      _e.displacementmap,
      _e.fog,
      {
        matcap: { value: null }
      }
    ]),
    vertexShader: je.meshmatcap_vert,
    fragmentShader: je.meshmatcap_frag
  },
  points: {
    uniforms: /* @__PURE__ */ zt([
      _e.points,
      _e.fog
    ]),
    vertexShader: je.points_vert,
    fragmentShader: je.points_frag
  },
  dashed: {
    uniforms: /* @__PURE__ */ zt([
      _e.common,
      _e.fog,
      {
        scale: { value: 1 },
        dashSize: { value: 1 },
        totalSize: { value: 2 }
      }
    ]),
    vertexShader: je.linedashed_vert,
    fragmentShader: je.linedashed_frag
  },
  depth: {
    uniforms: /* @__PURE__ */ zt([
      _e.common,
      _e.displacementmap
    ]),
    vertexShader: je.depth_vert,
    fragmentShader: je.depth_frag
  },
  normal: {
    uniforms: /* @__PURE__ */ zt([
      _e.common,
      _e.bumpmap,
      _e.normalmap,
      _e.displacementmap,
      {
        opacity: { value: 1 }
      }
    ]),
    vertexShader: je.meshnormal_vert,
    fragmentShader: je.meshnormal_frag
  },
  sprite: {
    uniforms: /* @__PURE__ */ zt([
      _e.sprite,
      _e.fog
    ]),
    vertexShader: je.sprite_vert,
    fragmentShader: je.sprite_frag
  },
  background: {
    uniforms: {
      uvTransform: { value: /* @__PURE__ */ new qe() },
      t2D: { value: null },
      backgroundIntensity: { value: 1 }
    },
    vertexShader: je.background_vert,
    fragmentShader: je.background_frag
  },
  backgroundCube: {
    uniforms: {
      envMap: { value: null },
      flipEnvMap: { value: -1 },
      backgroundBlurriness: { value: 0 },
      backgroundIntensity: { value: 1 },
      backgroundRotation: { value: /* @__PURE__ */ new qe() }
    },
    vertexShader: je.backgroundCube_vert,
    fragmentShader: je.backgroundCube_frag
  },
  cube: {
    uniforms: {
      tCube: { value: null },
      tFlip: { value: -1 },
      opacity: { value: 1 }
    },
    vertexShader: je.cube_vert,
    fragmentShader: je.cube_frag
  },
  equirect: {
    uniforms: {
      tEquirect: { value: null }
    },
    vertexShader: je.equirect_vert,
    fragmentShader: je.equirect_frag
  },
  distanceRGBA: {
    uniforms: /* @__PURE__ */ zt([
      _e.common,
      _e.displacementmap,
      {
        referencePosition: { value: /* @__PURE__ */ new N() },
        nearDistance: { value: 1 },
        farDistance: { value: 1e3 }
      }
    ]),
    vertexShader: je.distanceRGBA_vert,
    fragmentShader: je.distanceRGBA_frag
  },
  shadow: {
    uniforms: /* @__PURE__ */ zt([
      _e.lights,
      _e.fog,
      {
        color: { value: /* @__PURE__ */ new Xe(0) },
        opacity: { value: 1 }
      }
    ]),
    vertexShader: je.shadow_vert,
    fragmentShader: je.shadow_frag
  }
};
Ln.physical = {
  uniforms: /* @__PURE__ */ zt([
    Ln.standard.uniforms,
    {
      clearcoat: { value: 0 },
      clearcoatMap: { value: null },
      clearcoatMapTransform: { value: /* @__PURE__ */ new qe() },
      clearcoatNormalMap: { value: null },
      clearcoatNormalMapTransform: { value: /* @__PURE__ */ new qe() },
      clearcoatNormalScale: { value: /* @__PURE__ */ new Ve(1, 1) },
      clearcoatRoughness: { value: 0 },
      clearcoatRoughnessMap: { value: null },
      clearcoatRoughnessMapTransform: { value: /* @__PURE__ */ new qe() },
      dispersion: { value: 0 },
      iridescence: { value: 0 },
      iridescenceMap: { value: null },
      iridescenceMapTransform: { value: /* @__PURE__ */ new qe() },
      iridescenceIOR: { value: 1.3 },
      iridescenceThicknessMinimum: { value: 100 },
      iridescenceThicknessMaximum: { value: 400 },
      iridescenceThicknessMap: { value: null },
      iridescenceThicknessMapTransform: { value: /* @__PURE__ */ new qe() },
      sheen: { value: 0 },
      sheenColor: { value: /* @__PURE__ */ new Xe(0) },
      sheenColorMap: { value: null },
      sheenColorMapTransform: { value: /* @__PURE__ */ new qe() },
      sheenRoughness: { value: 1 },
      sheenRoughnessMap: { value: null },
      sheenRoughnessMapTransform: { value: /* @__PURE__ */ new qe() },
      transmission: { value: 0 },
      transmissionMap: { value: null },
      transmissionMapTransform: { value: /* @__PURE__ */ new qe() },
      transmissionSamplerSize: { value: /* @__PURE__ */ new Ve() },
      transmissionSamplerMap: { value: null },
      thickness: { value: 0 },
      thicknessMap: { value: null },
      thicknessMapTransform: { value: /* @__PURE__ */ new qe() },
      attenuationDistance: { value: 0 },
      attenuationColor: { value: /* @__PURE__ */ new Xe(0) },
      specularColor: { value: /* @__PURE__ */ new Xe(1, 1, 1) },
      specularColorMap: { value: null },
      specularColorMapTransform: { value: /* @__PURE__ */ new qe() },
      specularIntensity: { value: 1 },
      specularIntensityMap: { value: null },
      specularIntensityMapTransform: { value: /* @__PURE__ */ new qe() },
      anisotropyVector: { value: /* @__PURE__ */ new Ve() },
      anisotropyMap: { value: null },
      anisotropyMapTransform: { value: /* @__PURE__ */ new qe() }
    }
  ]),
  vertexShader: je.meshphysical_vert,
  fragmentShader: je.meshphysical_frag
};
const po = { r: 0, b: 0, g: 0 }, Ui = /* @__PURE__ */ new zn(), iM = /* @__PURE__ */ new pt();
function sM(n, e, t, i, s, r, o) {
  const a = new Xe(0);
  let l = r === !0 ? 0 : 1, c, u, h = null, f = 0, p = null;
  function v(A) {
    let M = A.isScene === !0 ? A.background : null;
    return M && M.isTexture && (M = (A.backgroundBlurriness > 0 ? t : e).get(M)), M;
  }
  function x(A) {
    let M = !1;
    const C = v(A);
    C === null ? d(a, l) : C && C.isColor && (d(C, 1), M = !0);
    const w = n.xr.getEnvironmentBlendMode();
    w === "additive" ? i.buffers.color.setClear(0, 0, 0, 1, o) : w === "alpha-blend" && i.buffers.color.setClear(0, 0, 0, 0, o), (n.autoClear || M) && (i.buffers.depth.setTest(!0), i.buffers.depth.setMask(!0), i.buffers.color.setMask(!0), n.clear(n.autoClearColor, n.autoClearDepth, n.autoClearStencil));
  }
  function m(A, M) {
    const C = v(M);
    C && (C.isCubeTexture || C.mapping === ta) ? (u === void 0 && (u = new vt(
      new Ki(1, 1, 1),
      new Ei({
        name: "BackgroundCubeMaterial",
        uniforms: zs(Ln.backgroundCube.uniforms),
        vertexShader: Ln.backgroundCube.vertexShader,
        fragmentShader: Ln.backgroundCube.fragmentShader,
        side: Wt,
        depthTest: !1,
        depthWrite: !1,
        fog: !1,
        allowOverride: !1
      })
    ), u.geometry.deleteAttribute("normal"), u.geometry.deleteAttribute("uv"), u.onBeforeRender = function(w, P, U) {
      this.matrixWorld.copyPosition(U.matrixWorld);
    }, Object.defineProperty(u.material, "envMap", {
      get: function() {
        return this.uniforms.envMap.value;
      }
    }), s.update(u)), Ui.copy(M.backgroundRotation), Ui.x *= -1, Ui.y *= -1, Ui.z *= -1, C.isCubeTexture && C.isRenderTargetTexture === !1 && (Ui.y *= -1, Ui.z *= -1), u.material.uniforms.envMap.value = C, u.material.uniforms.flipEnvMap.value = C.isCubeTexture && C.isRenderTargetTexture === !1 ? -1 : 1, u.material.uniforms.backgroundBlurriness.value = M.backgroundBlurriness, u.material.uniforms.backgroundIntensity.value = M.backgroundIntensity, u.material.uniforms.backgroundRotation.value.setFromMatrix4(iM.makeRotationFromEuler(Ui)), u.material.toneMapped = et.getTransfer(C.colorSpace) !== ot, (h !== C || f !== C.version || p !== n.toneMapping) && (u.material.needsUpdate = !0, h = C, f = C.version, p = n.toneMapping), u.layers.enableAll(), A.unshift(u, u.geometry, u.material, 0, 0, null)) : C && C.isTexture && (c === void 0 && (c = new vt(
      new Hs(2, 2),
      new Ei({
        name: "BackgroundMaterial",
        uniforms: zs(Ln.background.uniforms),
        vertexShader: Ln.background.vertexShader,
        fragmentShader: Ln.background.fragmentShader,
        side: yi,
        depthTest: !1,
        depthWrite: !1,
        fog: !1,
        allowOverride: !1
      })
    ), c.geometry.deleteAttribute("normal"), Object.defineProperty(c.material, "map", {
      get: function() {
        return this.uniforms.t2D.value;
      }
    }), s.update(c)), c.material.uniforms.t2D.value = C, c.material.uniforms.backgroundIntensity.value = M.backgroundIntensity, c.material.toneMapped = et.getTransfer(C.colorSpace) !== ot, C.matrixAutoUpdate === !0 && C.updateMatrix(), c.material.uniforms.uvTransform.value.copy(C.matrix), (h !== C || f !== C.version || p !== n.toneMapping) && (c.material.needsUpdate = !0, h = C, f = C.version, p = n.toneMapping), c.layers.enableAll(), A.unshift(c, c.geometry, c.material, 0, 0, null));
  }
  function d(A, M) {
    A.getRGB(po, Md(n)), i.buffers.color.setClear(po.r, po.g, po.b, M, o);
  }
  function b() {
    u !== void 0 && (u.geometry.dispose(), u.material.dispose(), u = void 0), c !== void 0 && (c.geometry.dispose(), c.material.dispose(), c = void 0);
  }
  return {
    getClearColor: function() {
      return a;
    },
    setClearColor: function(A, M = 1) {
      a.set(A), l = M, d(a, l);
    },
    getClearAlpha: function() {
      return l;
    },
    setClearAlpha: function(A) {
      l = A, d(a, l);
    },
    render: x,
    addToRenderList: m,
    dispose: b
  };
}
function rM(n, e) {
  const t = n.getParameter(n.MAX_VERTEX_ATTRIBS), i = {}, s = f(null);
  let r = s, o = !1;
  function a(y, D, L, V, Z) {
    let ne = !1;
    const J = h(V, L, D);
    r !== J && (r = J, c(r.object)), ne = p(y, V, L, Z), ne && v(y, V, L, Z), Z !== null && e.update(Z, n.ELEMENT_ARRAY_BUFFER), (ne || o) && (o = !1, M(y, D, L, V), Z !== null && n.bindBuffer(n.ELEMENT_ARRAY_BUFFER, e.get(Z).buffer));
  }
  function l() {
    return n.createVertexArray();
  }
  function c(y) {
    return n.bindVertexArray(y);
  }
  function u(y) {
    return n.deleteVertexArray(y);
  }
  function h(y, D, L) {
    const V = L.wireframe === !0;
    let Z = i[y.id];
    Z === void 0 && (Z = {}, i[y.id] = Z);
    let ne = Z[D.id];
    ne === void 0 && (ne = {}, Z[D.id] = ne);
    let J = ne[V];
    return J === void 0 && (J = f(l()), ne[V] = J), J;
  }
  function f(y) {
    const D = [], L = [], V = [];
    for (let Z = 0; Z < t; Z++)
      D[Z] = 0, L[Z] = 0, V[Z] = 0;
    return {
      // for backward compatibility on non-VAO support browser
      geometry: null,
      program: null,
      wireframe: !1,
      newAttributes: D,
      enabledAttributes: L,
      attributeDivisors: V,
      object: y,
      attributes: {},
      index: null
    };
  }
  function p(y, D, L, V) {
    const Z = r.attributes, ne = D.attributes;
    let J = 0;
    const ie = L.getAttributes();
    for (const H in ie)
      if (ie[H].location >= 0) {
        const ge = Z[H];
        let ye = ne[H];
        if (ye === void 0 && (H === "instanceMatrix" && y.instanceMatrix && (ye = y.instanceMatrix), H === "instanceColor" && y.instanceColor && (ye = y.instanceColor)), ge === void 0 || ge.attribute !== ye || ye && ge.data !== ye.data) return !0;
        J++;
      }
    return r.attributesNum !== J || r.index !== V;
  }
  function v(y, D, L, V) {
    const Z = {}, ne = D.attributes;
    let J = 0;
    const ie = L.getAttributes();
    for (const H in ie)
      if (ie[H].location >= 0) {
        let ge = ne[H];
        ge === void 0 && (H === "instanceMatrix" && y.instanceMatrix && (ge = y.instanceMatrix), H === "instanceColor" && y.instanceColor && (ge = y.instanceColor));
        const ye = {};
        ye.attribute = ge, ge && ge.data && (ye.data = ge.data), Z[H] = ye, J++;
      }
    r.attributes = Z, r.attributesNum = J, r.index = V;
  }
  function x() {
    const y = r.newAttributes;
    for (let D = 0, L = y.length; D < L; D++)
      y[D] = 0;
  }
  function m(y) {
    d(y, 0);
  }
  function d(y, D) {
    const L = r.newAttributes, V = r.enabledAttributes, Z = r.attributeDivisors;
    L[y] = 1, V[y] === 0 && (n.enableVertexAttribArray(y), V[y] = 1), Z[y] !== D && (n.vertexAttribDivisor(y, D), Z[y] = D);
  }
  function b() {
    const y = r.newAttributes, D = r.enabledAttributes;
    for (let L = 0, V = D.length; L < V; L++)
      D[L] !== y[L] && (n.disableVertexAttribArray(L), D[L] = 0);
  }
  function A(y, D, L, V, Z, ne, J) {
    J === !0 ? n.vertexAttribIPointer(y, D, L, Z, ne) : n.vertexAttribPointer(y, D, L, V, Z, ne);
  }
  function M(y, D, L, V) {
    x();
    const Z = V.attributes, ne = L.getAttributes(), J = D.defaultAttributeValues;
    for (const ie in ne) {
      const H = ne[ie];
      if (H.location >= 0) {
        let fe = Z[ie];
        if (fe === void 0 && (ie === "instanceMatrix" && y.instanceMatrix && (fe = y.instanceMatrix), ie === "instanceColor" && y.instanceColor && (fe = y.instanceColor)), fe !== void 0) {
          const ge = fe.normalized, ye = fe.itemSize, Fe = e.get(fe);
          if (Fe === void 0) continue;
          const Je = Fe.buffer, Ge = Fe.type, Ae = Fe.bytesPerElement, X = Ge === n.INT || Ge === n.UNSIGNED_INT || fe.gpuType === Mc;
          if (fe.isInterleavedBufferAttribute) {
            const re = fe.data, be = re.stride, Be = fe.offset;
            if (re.isInstancedInterleavedBuffer) {
              for (let Pe = 0; Pe < H.locationSize; Pe++)
                d(H.location + Pe, re.meshPerAttribute);
              y.isInstancedMesh !== !0 && V._maxInstanceCount === void 0 && (V._maxInstanceCount = re.meshPerAttribute * re.count);
            } else
              for (let Pe = 0; Pe < H.locationSize; Pe++)
                m(H.location + Pe);
            n.bindBuffer(n.ARRAY_BUFFER, Je);
            for (let Pe = 0; Pe < H.locationSize; Pe++)
              A(
                H.location + Pe,
                ye / H.locationSize,
                Ge,
                ge,
                be * Ae,
                (Be + ye / H.locationSize * Pe) * Ae,
                X
              );
          } else {
            if (fe.isInstancedBufferAttribute) {
              for (let re = 0; re < H.locationSize; re++)
                d(H.location + re, fe.meshPerAttribute);
              y.isInstancedMesh !== !0 && V._maxInstanceCount === void 0 && (V._maxInstanceCount = fe.meshPerAttribute * fe.count);
            } else
              for (let re = 0; re < H.locationSize; re++)
                m(H.location + re);
            n.bindBuffer(n.ARRAY_BUFFER, Je);
            for (let re = 0; re < H.locationSize; re++)
              A(
                H.location + re,
                ye / H.locationSize,
                Ge,
                ge,
                ye * Ae,
                ye / H.locationSize * re * Ae,
                X
              );
          }
        } else if (J !== void 0) {
          const ge = J[ie];
          if (ge !== void 0)
            switch (ge.length) {
              case 2:
                n.vertexAttrib2fv(H.location, ge);
                break;
              case 3:
                n.vertexAttrib3fv(H.location, ge);
                break;
              case 4:
                n.vertexAttrib4fv(H.location, ge);
                break;
              default:
                n.vertexAttrib1fv(H.location, ge);
            }
        }
      }
    }
    b();
  }
  function C() {
    U();
    for (const y in i) {
      const D = i[y];
      for (const L in D) {
        const V = D[L];
        for (const Z in V)
          u(V[Z].object), delete V[Z];
        delete D[L];
      }
      delete i[y];
    }
  }
  function w(y) {
    if (i[y.id] === void 0) return;
    const D = i[y.id];
    for (const L in D) {
      const V = D[L];
      for (const Z in V)
        u(V[Z].object), delete V[Z];
      delete D[L];
    }
    delete i[y.id];
  }
  function P(y) {
    for (const D in i) {
      const L = i[D];
      if (L[y.id] === void 0) continue;
      const V = L[y.id];
      for (const Z in V)
        u(V[Z].object), delete V[Z];
      delete L[y.id];
    }
  }
  function U() {
    S(), o = !0, r !== s && (r = s, c(r.object));
  }
  function S() {
    s.geometry = null, s.program = null, s.wireframe = !1;
  }
  return {
    setup: a,
    reset: U,
    resetDefaultState: S,
    dispose: C,
    releaseStatesOfGeometry: w,
    releaseStatesOfProgram: P,
    initAttributes: x,
    enableAttribute: m,
    disableUnusedAttributes: b
  };
}
function oM(n, e, t) {
  let i;
  function s(c) {
    i = c;
  }
  function r(c, u) {
    n.drawArrays(i, c, u), t.update(u, i, 1);
  }
  function o(c, u, h) {
    h !== 0 && (n.drawArraysInstanced(i, c, u, h), t.update(u, i, h));
  }
  function a(c, u, h) {
    if (h === 0) return;
    e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(i, c, 0, u, 0, h);
    let p = 0;
    for (let v = 0; v < h; v++)
      p += u[v];
    t.update(p, i, 1);
  }
  function l(c, u, h, f) {
    if (h === 0) return;
    const p = e.get("WEBGL_multi_draw");
    if (p === null)
      for (let v = 0; v < c.length; v++)
        o(c[v], u[v], f[v]);
    else {
      p.multiDrawArraysInstancedWEBGL(i, c, 0, u, 0, f, 0, h);
      let v = 0;
      for (let x = 0; x < h; x++)
        v += u[x] * f[x];
      t.update(v, i, 1);
    }
  }
  this.setMode = s, this.render = r, this.renderInstances = o, this.renderMultiDraw = a, this.renderMultiDrawInstances = l;
}
function aM(n, e, t, i) {
  let s;
  function r() {
    if (s !== void 0) return s;
    if (e.has("EXT_texture_filter_anisotropic") === !0) {
      const P = e.get("EXT_texture_filter_anisotropic");
      s = n.getParameter(P.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
    } else
      s = 0;
    return s;
  }
  function o(P) {
    return !(P !== xn && i.convert(P) !== n.getParameter(n.IMPLEMENTATION_COLOR_READ_FORMAT));
  }
  function a(P) {
    const U = P === Ir && (e.has("EXT_color_buffer_half_float") || e.has("EXT_color_buffer_float"));
    return !(P !== Bn && i.convert(P) !== n.getParameter(n.IMPLEMENTATION_COLOR_READ_TYPE) && // Edge and Chrome Mac < 52 (#9513)
    P !== ei && !U);
  }
  function l(P) {
    if (P === "highp") {
      if (n.getShaderPrecisionFormat(n.VERTEX_SHADER, n.HIGH_FLOAT).precision > 0 && n.getShaderPrecisionFormat(n.FRAGMENT_SHADER, n.HIGH_FLOAT).precision > 0)
        return "highp";
      P = "mediump";
    }
    return P === "mediump" && n.getShaderPrecisionFormat(n.VERTEX_SHADER, n.MEDIUM_FLOAT).precision > 0 && n.getShaderPrecisionFormat(n.FRAGMENT_SHADER, n.MEDIUM_FLOAT).precision > 0 ? "mediump" : "lowp";
  }
  let c = t.precision !== void 0 ? t.precision : "highp";
  const u = l(c);
  u !== c && (console.warn("THREE.WebGLRenderer:", c, "not supported, using", u, "instead."), c = u);
  const h = t.logarithmicDepthBuffer === !0, f = t.reversedDepthBuffer === !0 && e.has("EXT_clip_control"), p = n.getParameter(n.MAX_TEXTURE_IMAGE_UNITS), v = n.getParameter(n.MAX_VERTEX_TEXTURE_IMAGE_UNITS), x = n.getParameter(n.MAX_TEXTURE_SIZE), m = n.getParameter(n.MAX_CUBE_MAP_TEXTURE_SIZE), d = n.getParameter(n.MAX_VERTEX_ATTRIBS), b = n.getParameter(n.MAX_VERTEX_UNIFORM_VECTORS), A = n.getParameter(n.MAX_VARYING_VECTORS), M = n.getParameter(n.MAX_FRAGMENT_UNIFORM_VECTORS), C = v > 0, w = n.getParameter(n.MAX_SAMPLES);
  return {
    isWebGL2: !0,
    // keeping this for backwards compatibility
    getMaxAnisotropy: r,
    getMaxPrecision: l,
    textureFormatReadable: o,
    textureTypeReadable: a,
    precision: c,
    logarithmicDepthBuffer: h,
    reversedDepthBuffer: f,
    maxTextures: p,
    maxVertexTextures: v,
    maxTextureSize: x,
    maxCubemapSize: m,
    maxAttributes: d,
    maxVertexUniforms: b,
    maxVaryings: A,
    maxFragmentUniforms: M,
    vertexTextures: C,
    maxSamples: w
  };
}
function lM(n) {
  const e = this;
  let t = null, i = 0, s = !1, r = !1;
  const o = new mi(), a = new qe(), l = { value: null, needsUpdate: !1 };
  this.uniform = l, this.numPlanes = 0, this.numIntersection = 0, this.init = function(h, f) {
    const p = h.length !== 0 || f || // enable state of previous frame - the clipping code has to
    // run another frame in order to reset the state:
    i !== 0 || s;
    return s = f, i = h.length, p;
  }, this.beginShadows = function() {
    r = !0, u(null);
  }, this.endShadows = function() {
    r = !1;
  }, this.setGlobalState = function(h, f) {
    t = u(h, f, 0);
  }, this.setState = function(h, f, p) {
    const v = h.clippingPlanes, x = h.clipIntersection, m = h.clipShadows, d = n.get(h);
    if (!s || v === null || v.length === 0 || r && !m)
      r ? u(null) : c();
    else {
      const b = r ? 0 : i, A = b * 4;
      let M = d.clippingState || null;
      l.value = M, M = u(v, f, A, p);
      for (let C = 0; C !== A; ++C)
        M[C] = t[C];
      d.clippingState = M, this.numIntersection = x ? this.numPlanes : 0, this.numPlanes += b;
    }
  };
  function c() {
    l.value !== t && (l.value = t, l.needsUpdate = i > 0), e.numPlanes = i, e.numIntersection = 0;
  }
  function u(h, f, p, v) {
    const x = h !== null ? h.length : 0;
    let m = null;
    if (x !== 0) {
      if (m = l.value, v !== !0 || m === null) {
        const d = p + x * 4, b = f.matrixWorldInverse;
        a.getNormalMatrix(b), (m === null || m.length < d) && (m = new Float32Array(d));
        for (let A = 0, M = p; A !== x; ++A, M += 4)
          o.copy(h[A]).applyMatrix4(b, a), o.normal.toArray(m, M), m[M + 3] = o.constant;
      }
      l.value = m, l.needsUpdate = !0;
    }
    return e.numPlanes = x, e.numIntersection = 0, m;
  }
}
function cM(n) {
  let e = /* @__PURE__ */ new WeakMap();
  function t(o, a) {
    return a === Sl ? o.mapping = Fs : a === yl && (o.mapping = Os), o;
  }
  function i(o) {
    if (o && o.isTexture) {
      const a = o.mapping;
      if (a === Sl || a === yl)
        if (e.has(o)) {
          const l = e.get(o).texture;
          return t(l, o.mapping);
        } else {
          const l = o.image;
          if (l && l.height > 0) {
            const c = new i0(l.height);
            return c.fromEquirectangularTexture(n, o), e.set(o, c), o.addEventListener("dispose", s), t(c.texture, o.mapping);
          } else
            return null;
        }
    }
    return o;
  }
  function s(o) {
    const a = o.target;
    a.removeEventListener("dispose", s);
    const l = e.get(a);
    l !== void 0 && (e.delete(a), l.dispose());
  }
  function r() {
    e = /* @__PURE__ */ new WeakMap();
  }
  return {
    get: i,
    dispose: r
  };
}
const bs = 4, ch = [0.125, 0.215, 0.35, 0.446, 0.526, 0.582], Hi = 20, qa = /* @__PURE__ */ new Rd(), uh = /* @__PURE__ */ new Xe();
let ja = null, Ka = 0, $a = 0, Za = !1;
const Oi = (1 + Math.sqrt(5)) / 2, gs = 1 / Oi, hh = [
  /* @__PURE__ */ new N(-Oi, gs, 0),
  /* @__PURE__ */ new N(Oi, gs, 0),
  /* @__PURE__ */ new N(-gs, 0, Oi),
  /* @__PURE__ */ new N(gs, 0, Oi),
  /* @__PURE__ */ new N(0, Oi, -gs),
  /* @__PURE__ */ new N(0, Oi, gs),
  /* @__PURE__ */ new N(-1, 1, -1),
  /* @__PURE__ */ new N(1, 1, -1),
  /* @__PURE__ */ new N(-1, 1, 1),
  /* @__PURE__ */ new N(1, 1, 1)
], uM = /* @__PURE__ */ new N();
class fh {
  /**
   * Constructs a new PMREM generator.
   *
   * @param {WebGLRenderer} renderer - The renderer.
   */
  constructor(e) {
    this._renderer = e, this._pingPongRenderTarget = null, this._lodMax = 0, this._cubeSize = 0, this._lodPlanes = [], this._sizeLods = [], this._sigmas = [], this._blurMaterial = null, this._cubemapMaterial = null, this._equirectMaterial = null, this._compileMaterial(this._blurMaterial);
  }
  /**
   * Generates a PMREM from a supplied Scene, which can be faster than using an
   * image if networking bandwidth is low. Optional sigma specifies a blur radius
   * in radians to be applied to the scene before PMREM generation. Optional near
   * and far planes ensure the scene is rendered in its entirety.
   *
   * @param {Scene} scene - The scene to be captured.
   * @param {number} [sigma=0] - The blur radius in radians.
   * @param {number} [near=0.1] - The near plane distance.
   * @param {number} [far=100] - The far plane distance.
   * @param {Object} [options={}] - The configuration options.
   * @param {number} [options.size=256] - The texture size of the PMREM.
   * @param {Vector3} [options.renderTarget=origin] - The position of the internal cube camera that renders the scene.
   * @return {WebGLRenderTarget} The resulting PMREM.
   */
  fromScene(e, t = 0, i = 0.1, s = 100, r = {}) {
    const {
      size: o = 256,
      position: a = uM
    } = r;
    ja = this._renderer.getRenderTarget(), Ka = this._renderer.getActiveCubeFace(), $a = this._renderer.getActiveMipmapLevel(), Za = this._renderer.xr.enabled, this._renderer.xr.enabled = !1, this._setSize(o);
    const l = this._allocateTargets();
    return l.depthBuffer = !0, this._sceneToCubeUV(e, i, s, l, a), t > 0 && this._blur(l, 0, 0, t), this._applyPMREM(l), this._cleanup(l), l;
  }
  /**
   * Generates a PMREM from an equirectangular texture, which can be either LDR
   * or HDR. The ideal input image size is 1k (1024 x 512),
   * as this matches best with the 256 x 256 cubemap output.
   *
   * @param {Texture} equirectangular - The equirectangular texture to be converted.
   * @param {?WebGLRenderTarget} [renderTarget=null] - The render target to use.
   * @return {WebGLRenderTarget} The resulting PMREM.
   */
  fromEquirectangular(e, t = null) {
    return this._fromTexture(e, t);
  }
  /**
   * Generates a PMREM from an cubemap texture, which can be either LDR
   * or HDR. The ideal input cube size is 256 x 256,
   * as this matches best with the 256 x 256 cubemap output.
   *
   * @param {Texture} cubemap - The cubemap texture to be converted.
   * @param {?WebGLRenderTarget} [renderTarget=null] - The render target to use.
   * @return {WebGLRenderTarget} The resulting PMREM.
   */
  fromCubemap(e, t = null) {
    return this._fromTexture(e, t);
  }
  /**
   * Pre-compiles the cubemap shader. You can get faster start-up by invoking this method during
   * your texture's network fetch for increased concurrency.
   */
  compileCubemapShader() {
    this._cubemapMaterial === null && (this._cubemapMaterial = mh(), this._compileMaterial(this._cubemapMaterial));
  }
  /**
   * Pre-compiles the equirectangular shader. You can get faster start-up by invoking this method during
   * your texture's network fetch for increased concurrency.
   */
  compileEquirectangularShader() {
    this._equirectMaterial === null && (this._equirectMaterial = ph(), this._compileMaterial(this._equirectMaterial));
  }
  /**
   * Disposes of the PMREMGenerator's internal memory. Note that PMREMGenerator is a static class,
   * so you should not need more than one PMREMGenerator object. If you do, calling dispose() on
   * one of them will cause any others to also become unusable.
   */
  dispose() {
    this._dispose(), this._cubemapMaterial !== null && this._cubemapMaterial.dispose(), this._equirectMaterial !== null && this._equirectMaterial.dispose();
  }
  // private interface
  _setSize(e) {
    this._lodMax = Math.floor(Math.log2(e)), this._cubeSize = Math.pow(2, this._lodMax);
  }
  _dispose() {
    this._blurMaterial !== null && this._blurMaterial.dispose(), this._pingPongRenderTarget !== null && this._pingPongRenderTarget.dispose();
    for (let e = 0; e < this._lodPlanes.length; e++)
      this._lodPlanes[e].dispose();
  }
  _cleanup(e) {
    this._renderer.setRenderTarget(ja, Ka, $a), this._renderer.xr.enabled = Za, e.scissorTest = !1, mo(e, 0, 0, e.width, e.height);
  }
  _fromTexture(e, t) {
    e.mapping === Fs || e.mapping === Os ? this._setSize(e.image.length === 0 ? 16 : e.image[0].width || e.image[0].image.width) : this._setSize(e.image.width / 4), ja = this._renderer.getRenderTarget(), Ka = this._renderer.getActiveCubeFace(), $a = this._renderer.getActiveMipmapLevel(), Za = this._renderer.xr.enabled, this._renderer.xr.enabled = !1;
    const i = t || this._allocateTargets();
    return this._textureToCubeUV(e, i), this._applyPMREM(i), this._cleanup(i), i;
  }
  _allocateTargets() {
    const e = 3 * Math.max(this._cubeSize, 112), t = 4 * this._cubeSize, i = {
      magFilter: Un,
      minFilter: Un,
      generateMipmaps: !1,
      type: Ir,
      format: xn,
      colorSpace: Bs,
      depthBuffer: !1
    }, s = dh(e, t, i);
    if (this._pingPongRenderTarget === null || this._pingPongRenderTarget.width !== e || this._pingPongRenderTarget.height !== t) {
      this._pingPongRenderTarget !== null && this._dispose(), this._pingPongRenderTarget = dh(e, t, i);
      const { _lodMax: r } = this;
      ({ sizeLods: this._sizeLods, lodPlanes: this._lodPlanes, sigmas: this._sigmas } = hM(r)), this._blurMaterial = fM(r, e, t);
    }
    return s;
  }
  _compileMaterial(e) {
    const t = new vt(this._lodPlanes[0], e);
    this._renderer.compile(t, qa);
  }
  _sceneToCubeUV(e, t, i, s, r) {
    const l = new rn(90, 1, t, i), c = [1, -1, 1, 1, 1, 1], u = [1, 1, 1, -1, -1, -1], h = this._renderer, f = h.autoClear, p = h.toneMapping;
    h.getClearColor(uh), h.toneMapping = Mi, h.autoClear = !1, h.state.buffers.depth.getReversed() && (h.setRenderTarget(s), h.clearDepth(), h.setRenderTarget(null));
    const x = new Rn({
      name: "PMREM.Background",
      side: Wt,
      depthWrite: !1,
      depthTest: !1
    }), m = new vt(new Ki(), x);
    let d = !1;
    const b = e.background;
    b ? b.isColor && (x.color.copy(b), e.background = null, d = !0) : (x.color.copy(uh), d = !0);
    for (let A = 0; A < 6; A++) {
      const M = A % 3;
      M === 0 ? (l.up.set(0, c[A], 0), l.position.set(r.x, r.y, r.z), l.lookAt(r.x + u[A], r.y, r.z)) : M === 1 ? (l.up.set(0, 0, c[A]), l.position.set(r.x, r.y, r.z), l.lookAt(r.x, r.y + u[A], r.z)) : (l.up.set(0, c[A], 0), l.position.set(r.x, r.y, r.z), l.lookAt(r.x, r.y, r.z + u[A]));
      const C = this._cubeSize;
      mo(s, M * C, A > 2 ? C : 0, C, C), h.setRenderTarget(s), d && h.render(m, l), h.render(e, l);
    }
    m.geometry.dispose(), m.material.dispose(), h.toneMapping = p, h.autoClear = f, e.background = b;
  }
  _textureToCubeUV(e, t) {
    const i = this._renderer, s = e.mapping === Fs || e.mapping === Os;
    s ? (this._cubemapMaterial === null && (this._cubemapMaterial = mh()), this._cubemapMaterial.uniforms.flipEnvMap.value = e.isRenderTargetTexture === !1 ? -1 : 1) : this._equirectMaterial === null && (this._equirectMaterial = ph());
    const r = s ? this._cubemapMaterial : this._equirectMaterial, o = new vt(this._lodPlanes[0], r), a = r.uniforms;
    a.envMap.value = e;
    const l = this._cubeSize;
    mo(t, 0, 0, 3 * l, 2 * l), i.setRenderTarget(t), i.render(o, qa);
  }
  _applyPMREM(e) {
    const t = this._renderer, i = t.autoClear;
    t.autoClear = !1;
    const s = this._lodPlanes.length;
    for (let r = 1; r < s; r++) {
      const o = Math.sqrt(this._sigmas[r] * this._sigmas[r] - this._sigmas[r - 1] * this._sigmas[r - 1]), a = hh[(s - r - 1) % hh.length];
      this._blur(e, r - 1, r, o, a);
    }
    t.autoClear = i;
  }
  /**
   * This is a two-pass Gaussian blur for a cubemap. Normally this is done
   * vertically and horizontally, but this breaks down on a cube. Here we apply
   * the blur latitudinally (around the poles), and then longitudinally (towards
   * the poles) to approximate the orthogonally-separable blur. It is least
   * accurate at the poles, but still does a decent job.
   *
   * @private
   * @param {WebGLRenderTarget} cubeUVRenderTarget
   * @param {number} lodIn
   * @param {number} lodOut
   * @param {number} sigma
   * @param {Vector3} [poleAxis]
   */
  _blur(e, t, i, s, r) {
    const o = this._pingPongRenderTarget;
    this._halfBlur(
      e,
      o,
      t,
      i,
      s,
      "latitudinal",
      r
    ), this._halfBlur(
      o,
      e,
      i,
      i,
      s,
      "longitudinal",
      r
    );
  }
  _halfBlur(e, t, i, s, r, o, a) {
    const l = this._renderer, c = this._blurMaterial;
    o !== "latitudinal" && o !== "longitudinal" && console.error(
      "blur direction must be either latitudinal or longitudinal!"
    );
    const u = 3, h = new vt(this._lodPlanes[s], c), f = c.uniforms, p = this._sizeLods[i] - 1, v = isFinite(r) ? Math.PI / (2 * p) : 2 * Math.PI / (2 * Hi - 1), x = r / v, m = isFinite(r) ? 1 + Math.floor(u * x) : Hi;
    m > Hi && console.warn(`sigmaRadians, ${r}, is too large and will clip, as it requested ${m} samples when the maximum is set to ${Hi}`);
    const d = [];
    let b = 0;
    for (let P = 0; P < Hi; ++P) {
      const U = P / x, S = Math.exp(-U * U / 2);
      d.push(S), P === 0 ? b += S : P < m && (b += 2 * S);
    }
    for (let P = 0; P < d.length; P++)
      d[P] = d[P] / b;
    f.envMap.value = e.texture, f.samples.value = m, f.weights.value = d, f.latitudinal.value = o === "latitudinal", a && (f.poleAxis.value = a);
    const { _lodMax: A } = this;
    f.dTheta.value = v, f.mipInt.value = A - i;
    const M = this._sizeLods[s], C = 3 * M * (s > A - bs ? s - A + bs : 0), w = 4 * (this._cubeSize - M);
    mo(t, C, w, 3 * M, 2 * M), l.setRenderTarget(t), l.render(h, qa);
  }
}
function hM(n) {
  const e = [], t = [], i = [];
  let s = n;
  const r = n - bs + 1 + ch.length;
  for (let o = 0; o < r; o++) {
    const a = Math.pow(2, s);
    t.push(a);
    let l = 1 / a;
    o > n - bs ? l = ch[o - n + bs - 1] : o === 0 && (l = 0), i.push(l);
    const c = 1 / (a - 2), u = -c, h = 1 + c, f = [u, u, h, u, h, h, u, u, h, h, u, h], p = 6, v = 6, x = 3, m = 2, d = 1, b = new Float32Array(x * v * p), A = new Float32Array(m * v * p), M = new Float32Array(d * v * p);
    for (let w = 0; w < p; w++) {
      const P = w % 3 * 2 / 3 - 1, U = w > 2 ? 0 : -1, S = [
        P,
        U,
        0,
        P + 2 / 3,
        U,
        0,
        P + 2 / 3,
        U + 1,
        0,
        P,
        U,
        0,
        P + 2 / 3,
        U + 1,
        0,
        P,
        U + 1,
        0
      ];
      b.set(S, x * v * w), A.set(f, m * v * w);
      const y = [w, w, w, w, w, w];
      M.set(y, d * v * w);
    }
    const C = new Nt();
    C.setAttribute("position", new En(b, x)), C.setAttribute("uv", new En(A, m)), C.setAttribute("faceIndex", new En(M, d)), e.push(C), s > bs && s--;
  }
  return { lodPlanes: e, sizeLods: t, sigmas: i };
}
function dh(n, e, t) {
  const i = new ji(n, e, t);
  return i.texture.mapping = ta, i.texture.name = "PMREM.cubeUv", i.scissorTest = !0, i;
}
function mo(n, e, t, i, s) {
  n.viewport.set(e, t, i, s), n.scissor.set(e, t, i, s);
}
function fM(n, e, t) {
  const i = new Float32Array(Hi), s = new N(0, 1, 0);
  return new Ei({
    name: "SphericalGaussianBlur",
    defines: {
      n: Hi,
      CUBEUV_TEXEL_WIDTH: 1 / e,
      CUBEUV_TEXEL_HEIGHT: 1 / t,
      CUBEUV_MAX_MIP: `${n}.0`
    },
    uniforms: {
      envMap: { value: null },
      samples: { value: 1 },
      weights: { value: i },
      latitudinal: { value: !1 },
      dTheta: { value: 0 },
      mipInt: { value: 0 },
      poleAxis: { value: s }
    },
    vertexShader: Ic(),
    fragmentShader: (
      /* glsl */
      `

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`
    ),
    blending: xi,
    depthTest: !1,
    depthWrite: !1
  });
}
function ph() {
  return new Ei({
    name: "EquirectangularToCubeUV",
    uniforms: {
      envMap: { value: null }
    },
    vertexShader: Ic(),
    fragmentShader: (
      /* glsl */
      `

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`
    ),
    blending: xi,
    depthTest: !1,
    depthWrite: !1
  });
}
function mh() {
  return new Ei({
    name: "CubemapToCubeUV",
    uniforms: {
      envMap: { value: null },
      flipEnvMap: { value: -1 }
    },
    vertexShader: Ic(),
    fragmentShader: (
      /* glsl */
      `

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`
    ),
    blending: xi,
    depthTest: !1,
    depthWrite: !1
  });
}
function Ic() {
  return (
    /* glsl */
    `

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`
  );
}
function dM(n) {
  let e = /* @__PURE__ */ new WeakMap(), t = null;
  function i(a) {
    if (a && a.isTexture) {
      const l = a.mapping, c = l === Sl || l === yl, u = l === Fs || l === Os;
      if (c || u) {
        let h = e.get(a);
        const f = h !== void 0 ? h.texture.pmremVersion : 0;
        if (a.isRenderTargetTexture && a.pmremVersion !== f)
          return t === null && (t = new fh(n)), h = c ? t.fromEquirectangular(a, h) : t.fromCubemap(a, h), h.texture.pmremVersion = a.pmremVersion, e.set(a, h), h.texture;
        if (h !== void 0)
          return h.texture;
        {
          const p = a.image;
          return c && p && p.height > 0 || u && p && s(p) ? (t === null && (t = new fh(n)), h = c ? t.fromEquirectangular(a) : t.fromCubemap(a), h.texture.pmremVersion = a.pmremVersion, e.set(a, h), a.addEventListener("dispose", r), h.texture) : null;
        }
      }
    }
    return a;
  }
  function s(a) {
    let l = 0;
    const c = 6;
    for (let u = 0; u < c; u++)
      a[u] !== void 0 && l++;
    return l === c;
  }
  function r(a) {
    const l = a.target;
    l.removeEventListener("dispose", r);
    const c = e.get(l);
    c !== void 0 && (e.delete(l), c.dispose());
  }
  function o() {
    e = /* @__PURE__ */ new WeakMap(), t !== null && (t.dispose(), t = null);
  }
  return {
    get: i,
    dispose: o
  };
}
function pM(n) {
  const e = {};
  function t(i) {
    if (e[i] !== void 0)
      return e[i];
    let s;
    switch (i) {
      case "WEBGL_depth_texture":
        s = n.getExtension("WEBGL_depth_texture") || n.getExtension("MOZ_WEBGL_depth_texture") || n.getExtension("WEBKIT_WEBGL_depth_texture");
        break;
      case "EXT_texture_filter_anisotropic":
        s = n.getExtension("EXT_texture_filter_anisotropic") || n.getExtension("MOZ_EXT_texture_filter_anisotropic") || n.getExtension("WEBKIT_EXT_texture_filter_anisotropic");
        break;
      case "WEBGL_compressed_texture_s3tc":
        s = n.getExtension("WEBGL_compressed_texture_s3tc") || n.getExtension("MOZ_WEBGL_compressed_texture_s3tc") || n.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");
        break;
      case "WEBGL_compressed_texture_pvrtc":
        s = n.getExtension("WEBGL_compressed_texture_pvrtc") || n.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");
        break;
      default:
        s = n.getExtension(i);
    }
    return e[i] = s, s;
  }
  return {
    has: function(i) {
      return t(i) !== null;
    },
    init: function() {
      t("EXT_color_buffer_float"), t("WEBGL_clip_cull_distance"), t("OES_texture_float_linear"), t("EXT_color_buffer_half_float"), t("WEBGL_multisampled_render_to_texture"), t("WEBGL_render_shared_exponent");
    },
    get: function(i) {
      const s = t(i);
      return s === null && Cr("THREE.WebGLRenderer: " + i + " extension not supported."), s;
    }
  };
}
function mM(n, e, t, i) {
  const s = {}, r = /* @__PURE__ */ new WeakMap();
  function o(h) {
    const f = h.target;
    f.index !== null && e.remove(f.index);
    for (const v in f.attributes)
      e.remove(f.attributes[v]);
    f.removeEventListener("dispose", o), delete s[f.id];
    const p = r.get(f);
    p && (e.remove(p), r.delete(f)), i.releaseStatesOfGeometry(f), f.isInstancedBufferGeometry === !0 && delete f._maxInstanceCount, t.memory.geometries--;
  }
  function a(h, f) {
    return s[f.id] === !0 || (f.addEventListener("dispose", o), s[f.id] = !0, t.memory.geometries++), f;
  }
  function l(h) {
    const f = h.attributes;
    for (const p in f)
      e.update(f[p], n.ARRAY_BUFFER);
  }
  function c(h) {
    const f = [], p = h.index, v = h.attributes.position;
    let x = 0;
    if (p !== null) {
      const b = p.array;
      x = p.version;
      for (let A = 0, M = b.length; A < M; A += 3) {
        const C = b[A + 0], w = b[A + 1], P = b[A + 2];
        f.push(C, w, w, P, P, C);
      }
    } else if (v !== void 0) {
      const b = v.array;
      x = v.version;
      for (let A = 0, M = b.length / 3 - 1; A < M; A += 3) {
        const C = A + 0, w = A + 1, P = A + 2;
        f.push(C, w, w, P, P, C);
      }
    } else
      return;
    const m = new (pd(f) ? xd : vd)(f, 1);
    m.version = x;
    const d = r.get(h);
    d && e.remove(d), r.set(h, m);
  }
  function u(h) {
    const f = r.get(h);
    if (f) {
      const p = h.index;
      p !== null && f.version < p.version && c(h);
    } else
      c(h);
    return r.get(h);
  }
  return {
    get: a,
    update: l,
    getWireframeAttribute: u
  };
}
function _M(n, e, t) {
  let i;
  function s(f) {
    i = f;
  }
  let r, o;
  function a(f) {
    r = f.type, o = f.bytesPerElement;
  }
  function l(f, p) {
    n.drawElements(i, p, r, f * o), t.update(p, i, 1);
  }
  function c(f, p, v) {
    v !== 0 && (n.drawElementsInstanced(i, p, r, f * o, v), t.update(p, i, v));
  }
  function u(f, p, v) {
    if (v === 0) return;
    e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(i, p, 0, r, f, 0, v);
    let m = 0;
    for (let d = 0; d < v; d++)
      m += p[d];
    t.update(m, i, 1);
  }
  function h(f, p, v, x) {
    if (v === 0) return;
    const m = e.get("WEBGL_multi_draw");
    if (m === null)
      for (let d = 0; d < f.length; d++)
        c(f[d] / o, p[d], x[d]);
    else {
      m.multiDrawElementsInstancedWEBGL(i, p, 0, r, f, 0, x, 0, v);
      let d = 0;
      for (let b = 0; b < v; b++)
        d += p[b] * x[b];
      t.update(d, i, 1);
    }
  }
  this.setMode = s, this.setIndex = a, this.render = l, this.renderInstances = c, this.renderMultiDraw = u, this.renderMultiDrawInstances = h;
}
function gM(n) {
  const e = {
    geometries: 0,
    textures: 0
  }, t = {
    frame: 0,
    calls: 0,
    triangles: 0,
    points: 0,
    lines: 0
  };
  function i(r, o, a) {
    switch (t.calls++, o) {
      case n.TRIANGLES:
        t.triangles += a * (r / 3);
        break;
      case n.LINES:
        t.lines += a * (r / 2);
        break;
      case n.LINE_STRIP:
        t.lines += a * (r - 1);
        break;
      case n.LINE_LOOP:
        t.lines += a * r;
        break;
      case n.POINTS:
        t.points += a * r;
        break;
      default:
        console.error("THREE.WebGLInfo: Unknown draw mode:", o);
        break;
    }
  }
  function s() {
    t.calls = 0, t.triangles = 0, t.points = 0, t.lines = 0;
  }
  return {
    memory: e,
    render: t,
    programs: null,
    autoReset: !0,
    reset: s,
    update: i
  };
}
function vM(n, e, t) {
  const i = /* @__PURE__ */ new WeakMap(), s = new lt();
  function r(o, a, l) {
    const c = o.morphTargetInfluences, u = a.morphAttributes.position || a.morphAttributes.normal || a.morphAttributes.color, h = u !== void 0 ? u.length : 0;
    let f = i.get(a);
    if (f === void 0 || f.count !== h) {
      let S = function() {
        P.dispose(), i.delete(a), a.removeEventListener("dispose", S);
      };
      f !== void 0 && f.texture.dispose();
      const p = a.morphAttributes.position !== void 0, v = a.morphAttributes.normal !== void 0, x = a.morphAttributes.color !== void 0, m = a.morphAttributes.position || [], d = a.morphAttributes.normal || [], b = a.morphAttributes.color || [];
      let A = 0;
      p === !0 && (A = 1), v === !0 && (A = 2), x === !0 && (A = 3);
      let M = a.attributes.position.count * A, C = 1;
      M > e.maxTextureSize && (C = Math.ceil(M / e.maxTextureSize), M = e.maxTextureSize);
      const w = new Float32Array(M * C * 4 * h), P = new md(w, M, C, h);
      P.type = ei, P.needsUpdate = !0;
      const U = A * 4;
      for (let y = 0; y < h; y++) {
        const D = m[y], L = d[y], V = b[y], Z = M * C * 4 * y;
        for (let ne = 0; ne < D.count; ne++) {
          const J = ne * U;
          p === !0 && (s.fromBufferAttribute(D, ne), w[Z + J + 0] = s.x, w[Z + J + 1] = s.y, w[Z + J + 2] = s.z, w[Z + J + 3] = 0), v === !0 && (s.fromBufferAttribute(L, ne), w[Z + J + 4] = s.x, w[Z + J + 5] = s.y, w[Z + J + 6] = s.z, w[Z + J + 7] = 0), x === !0 && (s.fromBufferAttribute(V, ne), w[Z + J + 8] = s.x, w[Z + J + 9] = s.y, w[Z + J + 10] = s.z, w[Z + J + 11] = V.itemSize === 4 ? s.w : 1);
        }
      }
      f = {
        count: h,
        texture: P,
        size: new Ve(M, C)
      }, i.set(a, f), a.addEventListener("dispose", S);
    }
    if (o.isInstancedMesh === !0 && o.morphTexture !== null)
      l.getUniforms().setValue(n, "morphTexture", o.morphTexture, t);
    else {
      let p = 0;
      for (let x = 0; x < c.length; x++)
        p += c[x];
      const v = a.morphTargetsRelative ? 1 : 1 - p;
      l.getUniforms().setValue(n, "morphTargetBaseInfluence", v), l.getUniforms().setValue(n, "morphTargetInfluences", c);
    }
    l.getUniforms().setValue(n, "morphTargetsTexture", f.texture, t), l.getUniforms().setValue(n, "morphTargetsTextureSize", f.size);
  }
  return {
    update: r
  };
}
function xM(n, e, t, i) {
  let s = /* @__PURE__ */ new WeakMap();
  function r(l) {
    const c = i.render.frame, u = l.geometry, h = e.get(l, u);
    if (s.get(h) !== c && (e.update(h), s.set(h, c)), l.isInstancedMesh && (l.hasEventListener("dispose", a) === !1 && l.addEventListener("dispose", a), s.get(l) !== c && (t.update(l.instanceMatrix, n.ARRAY_BUFFER), l.instanceColor !== null && t.update(l.instanceColor, n.ARRAY_BUFFER), s.set(l, c))), l.isSkinnedMesh) {
      const f = l.skeleton;
      s.get(f) !== c && (f.update(), s.set(f, c));
    }
    return h;
  }
  function o() {
    s = /* @__PURE__ */ new WeakMap();
  }
  function a(l) {
    const c = l.target;
    c.removeEventListener("dispose", a), t.remove(c.instanceMatrix), c.instanceColor !== null && t.remove(c.instanceColor);
  }
  return {
    update: r,
    dispose: o
  };
}
const Pd = /* @__PURE__ */ new Zt(), _h = /* @__PURE__ */ new bd(1, 1), Dd = /* @__PURE__ */ new md(), Ld = /* @__PURE__ */ new Hg(), Id = /* @__PURE__ */ new yd(), gh = [], vh = [], xh = new Float32Array(16), Mh = new Float32Array(9), Sh = new Float32Array(4);
function ks(n, e, t) {
  const i = n[0];
  if (i <= 0 || i > 0) return n;
  const s = e * t;
  let r = gh[s];
  if (r === void 0 && (r = new Float32Array(s), gh[s] = r), e !== 0) {
    i.toArray(r, 0);
    for (let o = 1, a = 0; o !== e; ++o)
      a += t, n[o].toArray(r, a);
  }
  return r;
}
function bt(n, e) {
  if (n.length !== e.length) return !1;
  for (let t = 0, i = n.length; t < i; t++)
    if (n[t] !== e[t]) return !1;
  return !0;
}
function At(n, e) {
  for (let t = 0, i = e.length; t < i; t++)
    n[t] = e[t];
}
function ia(n, e) {
  let t = vh[e];
  t === void 0 && (t = new Int32Array(e), vh[e] = t);
  for (let i = 0; i !== e; ++i)
    t[i] = n.allocateTextureUnit();
  return t;
}
function MM(n, e) {
  const t = this.cache;
  t[0] !== e && (n.uniform1f(this.addr, e), t[0] = e);
}
function SM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y) && (n.uniform2f(this.addr, e.x, e.y), t[0] = e.x, t[1] = e.y);
  else {
    if (bt(t, e)) return;
    n.uniform2fv(this.addr, e), At(t, e);
  }
}
function yM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y || t[2] !== e.z) && (n.uniform3f(this.addr, e.x, e.y, e.z), t[0] = e.x, t[1] = e.y, t[2] = e.z);
  else if (e.r !== void 0)
    (t[0] !== e.r || t[1] !== e.g || t[2] !== e.b) && (n.uniform3f(this.addr, e.r, e.g, e.b), t[0] = e.r, t[1] = e.g, t[2] = e.b);
  else {
    if (bt(t, e)) return;
    n.uniform3fv(this.addr, e), At(t, e);
  }
}
function EM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y || t[2] !== e.z || t[3] !== e.w) && (n.uniform4f(this.addr, e.x, e.y, e.z, e.w), t[0] = e.x, t[1] = e.y, t[2] = e.z, t[3] = e.w);
  else {
    if (bt(t, e)) return;
    n.uniform4fv(this.addr, e), At(t, e);
  }
}
function TM(n, e) {
  const t = this.cache, i = e.elements;
  if (i === void 0) {
    if (bt(t, e)) return;
    n.uniformMatrix2fv(this.addr, !1, e), At(t, e);
  } else {
    if (bt(t, i)) return;
    Sh.set(i), n.uniformMatrix2fv(this.addr, !1, Sh), At(t, i);
  }
}
function bM(n, e) {
  const t = this.cache, i = e.elements;
  if (i === void 0) {
    if (bt(t, e)) return;
    n.uniformMatrix3fv(this.addr, !1, e), At(t, e);
  } else {
    if (bt(t, i)) return;
    Mh.set(i), n.uniformMatrix3fv(this.addr, !1, Mh), At(t, i);
  }
}
function AM(n, e) {
  const t = this.cache, i = e.elements;
  if (i === void 0) {
    if (bt(t, e)) return;
    n.uniformMatrix4fv(this.addr, !1, e), At(t, e);
  } else {
    if (bt(t, i)) return;
    xh.set(i), n.uniformMatrix4fv(this.addr, !1, xh), At(t, i);
  }
}
function wM(n, e) {
  const t = this.cache;
  t[0] !== e && (n.uniform1i(this.addr, e), t[0] = e);
}
function RM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y) && (n.uniform2i(this.addr, e.x, e.y), t[0] = e.x, t[1] = e.y);
  else {
    if (bt(t, e)) return;
    n.uniform2iv(this.addr, e), At(t, e);
  }
}
function CM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y || t[2] !== e.z) && (n.uniform3i(this.addr, e.x, e.y, e.z), t[0] = e.x, t[1] = e.y, t[2] = e.z);
  else {
    if (bt(t, e)) return;
    n.uniform3iv(this.addr, e), At(t, e);
  }
}
function PM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y || t[2] !== e.z || t[3] !== e.w) && (n.uniform4i(this.addr, e.x, e.y, e.z, e.w), t[0] = e.x, t[1] = e.y, t[2] = e.z, t[3] = e.w);
  else {
    if (bt(t, e)) return;
    n.uniform4iv(this.addr, e), At(t, e);
  }
}
function DM(n, e) {
  const t = this.cache;
  t[0] !== e && (n.uniform1ui(this.addr, e), t[0] = e);
}
function LM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y) && (n.uniform2ui(this.addr, e.x, e.y), t[0] = e.x, t[1] = e.y);
  else {
    if (bt(t, e)) return;
    n.uniform2uiv(this.addr, e), At(t, e);
  }
}
function IM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y || t[2] !== e.z) && (n.uniform3ui(this.addr, e.x, e.y, e.z), t[0] = e.x, t[1] = e.y, t[2] = e.z);
  else {
    if (bt(t, e)) return;
    n.uniform3uiv(this.addr, e), At(t, e);
  }
}
function UM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y || t[2] !== e.z || t[3] !== e.w) && (n.uniform4ui(this.addr, e.x, e.y, e.z, e.w), t[0] = e.x, t[1] = e.y, t[2] = e.z, t[3] = e.w);
  else {
    if (bt(t, e)) return;
    n.uniform4uiv(this.addr, e), At(t, e);
  }
}
function NM(n, e, t) {
  const i = this.cache, s = t.allocateTextureUnit();
  i[0] !== s && (n.uniform1i(this.addr, s), i[0] = s);
  let r;
  this.type === n.SAMPLER_2D_SHADOW ? (_h.compareFunction = dd, r = _h) : r = Pd, t.setTexture2D(e || r, s);
}
function FM(n, e, t) {
  const i = this.cache, s = t.allocateTextureUnit();
  i[0] !== s && (n.uniform1i(this.addr, s), i[0] = s), t.setTexture3D(e || Ld, s);
}
function OM(n, e, t) {
  const i = this.cache, s = t.allocateTextureUnit();
  i[0] !== s && (n.uniform1i(this.addr, s), i[0] = s), t.setTextureCube(e || Id, s);
}
function BM(n, e, t) {
  const i = this.cache, s = t.allocateTextureUnit();
  i[0] !== s && (n.uniform1i(this.addr, s), i[0] = s), t.setTexture2DArray(e || Dd, s);
}
function zM(n) {
  switch (n) {
    case 5126:
      return MM;
    // FLOAT
    case 35664:
      return SM;
    // _VEC2
    case 35665:
      return yM;
    // _VEC3
    case 35666:
      return EM;
    // _VEC4
    case 35674:
      return TM;
    // _MAT2
    case 35675:
      return bM;
    // _MAT3
    case 35676:
      return AM;
    // _MAT4
    case 5124:
    case 35670:
      return wM;
    // INT, BOOL
    case 35667:
    case 35671:
      return RM;
    // _VEC2
    case 35668:
    case 35672:
      return CM;
    // _VEC3
    case 35669:
    case 35673:
      return PM;
    // _VEC4
    case 5125:
      return DM;
    // UINT
    case 36294:
      return LM;
    // _VEC2
    case 36295:
      return IM;
    // _VEC3
    case 36296:
      return UM;
    // _VEC4
    case 35678:
    // SAMPLER_2D
    case 36198:
    // SAMPLER_EXTERNAL_OES
    case 36298:
    // INT_SAMPLER_2D
    case 36306:
    // UNSIGNED_INT_SAMPLER_2D
    case 35682:
      return NM;
    case 35679:
    // SAMPLER_3D
    case 36299:
    // INT_SAMPLER_3D
    case 36307:
      return FM;
    case 35680:
    // SAMPLER_CUBE
    case 36300:
    // INT_SAMPLER_CUBE
    case 36308:
    // UNSIGNED_INT_SAMPLER_CUBE
    case 36293:
      return OM;
    case 36289:
    // SAMPLER_2D_ARRAY
    case 36303:
    // INT_SAMPLER_2D_ARRAY
    case 36311:
    // UNSIGNED_INT_SAMPLER_2D_ARRAY
    case 36292:
      return BM;
  }
}
function HM(n, e) {
  n.uniform1fv(this.addr, e);
}
function VM(n, e) {
  const t = ks(e, this.size, 2);
  n.uniform2fv(this.addr, t);
}
function kM(n, e) {
  const t = ks(e, this.size, 3);
  n.uniform3fv(this.addr, t);
}
function GM(n, e) {
  const t = ks(e, this.size, 4);
  n.uniform4fv(this.addr, t);
}
function WM(n, e) {
  const t = ks(e, this.size, 4);
  n.uniformMatrix2fv(this.addr, !1, t);
}
function XM(n, e) {
  const t = ks(e, this.size, 9);
  n.uniformMatrix3fv(this.addr, !1, t);
}
function YM(n, e) {
  const t = ks(e, this.size, 16);
  n.uniformMatrix4fv(this.addr, !1, t);
}
function qM(n, e) {
  n.uniform1iv(this.addr, e);
}
function jM(n, e) {
  n.uniform2iv(this.addr, e);
}
function KM(n, e) {
  n.uniform3iv(this.addr, e);
}
function $M(n, e) {
  n.uniform4iv(this.addr, e);
}
function ZM(n, e) {
  n.uniform1uiv(this.addr, e);
}
function JM(n, e) {
  n.uniform2uiv(this.addr, e);
}
function QM(n, e) {
  n.uniform3uiv(this.addr, e);
}
function eS(n, e) {
  n.uniform4uiv(this.addr, e);
}
function tS(n, e, t) {
  const i = this.cache, s = e.length, r = ia(t, s);
  bt(i, r) || (n.uniform1iv(this.addr, r), At(i, r));
  for (let o = 0; o !== s; ++o)
    t.setTexture2D(e[o] || Pd, r[o]);
}
function nS(n, e, t) {
  const i = this.cache, s = e.length, r = ia(t, s);
  bt(i, r) || (n.uniform1iv(this.addr, r), At(i, r));
  for (let o = 0; o !== s; ++o)
    t.setTexture3D(e[o] || Ld, r[o]);
}
function iS(n, e, t) {
  const i = this.cache, s = e.length, r = ia(t, s);
  bt(i, r) || (n.uniform1iv(this.addr, r), At(i, r));
  for (let o = 0; o !== s; ++o)
    t.setTextureCube(e[o] || Id, r[o]);
}
function sS(n, e, t) {
  const i = this.cache, s = e.length, r = ia(t, s);
  bt(i, r) || (n.uniform1iv(this.addr, r), At(i, r));
  for (let o = 0; o !== s; ++o)
    t.setTexture2DArray(e[o] || Dd, r[o]);
}
function rS(n) {
  switch (n) {
    case 5126:
      return HM;
    // FLOAT
    case 35664:
      return VM;
    // _VEC2
    case 35665:
      return kM;
    // _VEC3
    case 35666:
      return GM;
    // _VEC4
    case 35674:
      return WM;
    // _MAT2
    case 35675:
      return XM;
    // _MAT3
    case 35676:
      return YM;
    // _MAT4
    case 5124:
    case 35670:
      return qM;
    // INT, BOOL
    case 35667:
    case 35671:
      return jM;
    // _VEC2
    case 35668:
    case 35672:
      return KM;
    // _VEC3
    case 35669:
    case 35673:
      return $M;
    // _VEC4
    case 5125:
      return ZM;
    // UINT
    case 36294:
      return JM;
    // _VEC2
    case 36295:
      return QM;
    // _VEC3
    case 36296:
      return eS;
    // _VEC4
    case 35678:
    // SAMPLER_2D
    case 36198:
    // SAMPLER_EXTERNAL_OES
    case 36298:
    // INT_SAMPLER_2D
    case 36306:
    // UNSIGNED_INT_SAMPLER_2D
    case 35682:
      return tS;
    case 35679:
    // SAMPLER_3D
    case 36299:
    // INT_SAMPLER_3D
    case 36307:
      return nS;
    case 35680:
    // SAMPLER_CUBE
    case 36300:
    // INT_SAMPLER_CUBE
    case 36308:
    // UNSIGNED_INT_SAMPLER_CUBE
    case 36293:
      return iS;
    case 36289:
    // SAMPLER_2D_ARRAY
    case 36303:
    // INT_SAMPLER_2D_ARRAY
    case 36311:
    // UNSIGNED_INT_SAMPLER_2D_ARRAY
    case 36292:
      return sS;
  }
}
class oS {
  constructor(e, t, i) {
    this.id = e, this.addr = i, this.cache = [], this.type = t.type, this.setValue = zM(t.type);
  }
}
class aS {
  constructor(e, t, i) {
    this.id = e, this.addr = i, this.cache = [], this.type = t.type, this.size = t.size, this.setValue = rS(t.type);
  }
}
class lS {
  constructor(e) {
    this.id = e, this.seq = [], this.map = {};
  }
  setValue(e, t, i) {
    const s = this.seq;
    for (let r = 0, o = s.length; r !== o; ++r) {
      const a = s[r];
      a.setValue(e, t[a.id], i);
    }
  }
}
const Ja = /(\w+)(\])?(\[|\.)?/g;
function yh(n, e) {
  n.seq.push(e), n.map[e.id] = e;
}
function cS(n, e, t) {
  const i = n.name, s = i.length;
  for (Ja.lastIndex = 0; ; ) {
    const r = Ja.exec(i), o = Ja.lastIndex;
    let a = r[1];
    const l = r[2] === "]", c = r[3];
    if (l && (a = a | 0), c === void 0 || c === "[" && o + 2 === s) {
      yh(t, c === void 0 ? new oS(a, n, e) : new aS(a, n, e));
      break;
    } else {
      let h = t.map[a];
      h === void 0 && (h = new lS(a), yh(t, h)), t = h;
    }
  }
}
class Ao {
  constructor(e, t) {
    this.seq = [], this.map = {};
    const i = e.getProgramParameter(t, e.ACTIVE_UNIFORMS);
    for (let s = 0; s < i; ++s) {
      const r = e.getActiveUniform(t, s), o = e.getUniformLocation(t, r.name);
      cS(r, o, this);
    }
  }
  setValue(e, t, i, s) {
    const r = this.map[t];
    r !== void 0 && r.setValue(e, i, s);
  }
  setOptional(e, t, i) {
    const s = t[i];
    s !== void 0 && this.setValue(e, i, s);
  }
  static upload(e, t, i, s) {
    for (let r = 0, o = t.length; r !== o; ++r) {
      const a = t[r], l = i[a.id];
      l.needsUpdate !== !1 && a.setValue(e, l.value, s);
    }
  }
  static seqWithValue(e, t) {
    const i = [];
    for (let s = 0, r = e.length; s !== r; ++s) {
      const o = e[s];
      o.id in t && i.push(o);
    }
    return i;
  }
}
function Eh(n, e, t) {
  const i = n.createShader(e);
  return n.shaderSource(i, t), n.compileShader(i), i;
}
const uS = 37297;
let hS = 0;
function fS(n, e) {
  const t = n.split(`
`), i = [], s = Math.max(e - 6, 0), r = Math.min(e + 6, t.length);
  for (let o = s; o < r; o++) {
    const a = o + 1;
    i.push(`${a === e ? ">" : " "} ${a}: ${t[o]}`);
  }
  return i.join(`
`);
}
const Th = /* @__PURE__ */ new qe();
function dS(n) {
  et._getMatrix(Th, et.workingColorSpace, n);
  const e = `mat3( ${Th.elements.map((t) => t.toFixed(4))} )`;
  switch (et.getTransfer(n)) {
    case Bo:
      return [e, "LinearTransferOETF"];
    case ot:
      return [e, "sRGBTransferOETF"];
    default:
      return console.warn("THREE.WebGLProgram: Unsupported color space: ", n), [e, "LinearTransferOETF"];
  }
}
function bh(n, e, t) {
  const i = n.getShaderParameter(e, n.COMPILE_STATUS), r = (n.getShaderInfoLog(e) || "").trim();
  if (i && r === "") return "";
  const o = /ERROR: 0:(\d+)/.exec(r);
  if (o) {
    const a = parseInt(o[1]);
    return t.toUpperCase() + `

` + r + `

` + fS(n.getShaderSource(e), a);
  } else
    return r;
}
function pS(n, e) {
  const t = dS(e);
  return [
    `vec4 ${n}( vec4 value ) {`,
    `	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,
    "}"
  ].join(`
`);
}
function mS(n, e) {
  let t;
  switch (e) {
    case pg:
      t = "Linear";
      break;
    case mg:
      t = "Reinhard";
      break;
    case _g:
      t = "Cineon";
      break;
    case nd:
      t = "ACESFilmic";
      break;
    case vg:
      t = "AgX";
      break;
    case xg:
      t = "Neutral";
      break;
    case gg:
      t = "Custom";
      break;
    default:
      console.warn("THREE.WebGLProgram: Unsupported toneMapping:", e), t = "Linear";
  }
  return "vec3 " + n + "( vec3 color ) { return " + t + "ToneMapping( color ); }";
}
const _o = /* @__PURE__ */ new N();
function _S() {
  et.getLuminanceCoefficients(_o);
  const n = _o.x.toFixed(4), e = _o.y.toFixed(4), t = _o.z.toFixed(4);
  return [
    "float luminance( const in vec3 rgb ) {",
    `	const vec3 weights = vec3( ${n}, ${e}, ${t} );`,
    "	return dot( weights, rgb );",
    "}"
  ].join(`
`);
}
function gS(n) {
  return [
    n.extensionClipCullDistance ? "#extension GL_ANGLE_clip_cull_distance : require" : "",
    n.extensionMultiDraw ? "#extension GL_ANGLE_multi_draw : require" : ""
  ].filter(or).join(`
`);
}
function vS(n) {
  const e = [];
  for (const t in n) {
    const i = n[t];
    i !== !1 && e.push("#define " + t + " " + i);
  }
  return e.join(`
`);
}
function xS(n, e) {
  const t = {}, i = n.getProgramParameter(e, n.ACTIVE_ATTRIBUTES);
  for (let s = 0; s < i; s++) {
    const r = n.getActiveAttrib(e, s), o = r.name;
    let a = 1;
    r.type === n.FLOAT_MAT2 && (a = 2), r.type === n.FLOAT_MAT3 && (a = 3), r.type === n.FLOAT_MAT4 && (a = 4), t[o] = {
      type: r.type,
      location: n.getAttribLocation(e, o),
      locationSize: a
    };
  }
  return t;
}
function or(n) {
  return n !== "";
}
function Ah(n, e) {
  const t = e.numSpotLightShadows + e.numSpotLightMaps - e.numSpotLightShadowsWithMaps;
  return n.replace(/NUM_DIR_LIGHTS/g, e.numDirLights).replace(/NUM_SPOT_LIGHTS/g, e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g, e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g, t).replace(/NUM_RECT_AREA_LIGHTS/g, e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g, e.numPointLights).replace(/NUM_HEMI_LIGHTS/g, e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g, e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g, e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g, e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g, e.numPointLightShadows);
}
function wh(n, e) {
  return n.replace(/NUM_CLIPPING_PLANES/g, e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g, e.numClippingPlanes - e.numClipIntersection);
}
const MS = /^[ \t]*#include +<([\w\d./]+)>/gm;
function tc(n) {
  return n.replace(MS, yS);
}
const SS = /* @__PURE__ */ new Map();
function yS(n, e) {
  let t = je[e];
  if (t === void 0) {
    const i = SS.get(e);
    if (i !== void 0)
      t = je[i], console.warn('THREE.WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.', e, i);
    else
      throw new Error("Can not resolve #include <" + e + ">");
  }
  return tc(t);
}
const ES = /#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;
function Rh(n) {
  return n.replace(ES, TS);
}
function TS(n, e, t, i) {
  let s = "";
  for (let r = parseInt(e); r < parseInt(t); r++)
    s += i.replace(/\[\s*i\s*\]/g, "[ " + r + " ]").replace(/UNROLLED_LOOP_INDEX/g, r);
  return s;
}
function Ch(n) {
  let e = `precision ${n.precision} float;
	precision ${n.precision} int;
	precision ${n.precision} sampler2D;
	precision ${n.precision} samplerCube;
	precision ${n.precision} sampler3D;
	precision ${n.precision} sampler2DArray;
	precision ${n.precision} sampler2DShadow;
	precision ${n.precision} samplerCubeShadow;
	precision ${n.precision} sampler2DArrayShadow;
	precision ${n.precision} isampler2D;
	precision ${n.precision} isampler3D;
	precision ${n.precision} isamplerCube;
	precision ${n.precision} isampler2DArray;
	precision ${n.precision} usampler2D;
	precision ${n.precision} usampler3D;
	precision ${n.precision} usamplerCube;
	precision ${n.precision} usampler2DArray;
	`;
  return n.precision === "highp" ? e += `
#define HIGH_PRECISION` : n.precision === "mediump" ? e += `
#define MEDIUM_PRECISION` : n.precision === "lowp" && (e += `
#define LOW_PRECISION`), e;
}
function bS(n) {
  let e = "SHADOWMAP_TYPE_BASIC";
  return n.shadowMapType === ed ? e = "SHADOWMAP_TYPE_PCF" : n.shadowMapType === q_ ? e = "SHADOWMAP_TYPE_PCF_SOFT" : n.shadowMapType === jn && (e = "SHADOWMAP_TYPE_VSM"), e;
}
function AS(n) {
  let e = "ENVMAP_TYPE_CUBE";
  if (n.envMap)
    switch (n.envMapMode) {
      case Fs:
      case Os:
        e = "ENVMAP_TYPE_CUBE";
        break;
      case ta:
        e = "ENVMAP_TYPE_CUBE_UV";
        break;
    }
  return e;
}
function wS(n) {
  let e = "ENVMAP_MODE_REFLECTION";
  return n.envMap && n.envMapMode === Os && (e = "ENVMAP_MODE_REFRACTION"), e;
}
function RS(n) {
  let e = "ENVMAP_BLENDING_NONE";
  if (n.envMap)
    switch (n.combine) {
      case td:
        e = "ENVMAP_BLENDING_MULTIPLY";
        break;
      case fg:
        e = "ENVMAP_BLENDING_MIX";
        break;
      case dg:
        e = "ENVMAP_BLENDING_ADD";
        break;
    }
  return e;
}
function CS(n) {
  const e = n.envMapCubeUVHeight;
  if (e === null) return null;
  const t = Math.log2(e) - 2, i = 1 / e;
  return { texelWidth: 1 / (3 * Math.max(Math.pow(2, t), 112)), texelHeight: i, maxMip: t };
}
function PS(n, e, t, i) {
  const s = n.getContext(), r = t.defines;
  let o = t.vertexShader, a = t.fragmentShader;
  const l = bS(t), c = AS(t), u = wS(t), h = RS(t), f = CS(t), p = gS(t), v = vS(r), x = s.createProgram();
  let m, d, b = t.glslVersion ? "#version " + t.glslVersion + `
` : "";
  t.isRawShaderMaterial ? (m = [
    "#define SHADER_TYPE " + t.shaderType,
    "#define SHADER_NAME " + t.shaderName,
    v
  ].filter(or).join(`
`), m.length > 0 && (m += `
`), d = [
    "#define SHADER_TYPE " + t.shaderType,
    "#define SHADER_NAME " + t.shaderName,
    v
  ].filter(or).join(`
`), d.length > 0 && (d += `
`)) : (m = [
    Ch(t),
    "#define SHADER_TYPE " + t.shaderType,
    "#define SHADER_NAME " + t.shaderName,
    v,
    t.extensionClipCullDistance ? "#define USE_CLIP_DISTANCE" : "",
    t.batching ? "#define USE_BATCHING" : "",
    t.batchingColor ? "#define USE_BATCHING_COLOR" : "",
    t.instancing ? "#define USE_INSTANCING" : "",
    t.instancingColor ? "#define USE_INSTANCING_COLOR" : "",
    t.instancingMorph ? "#define USE_INSTANCING_MORPH" : "",
    t.useFog && t.fog ? "#define USE_FOG" : "",
    t.useFog && t.fogExp2 ? "#define FOG_EXP2" : "",
    t.map ? "#define USE_MAP" : "",
    t.envMap ? "#define USE_ENVMAP" : "",
    t.envMap ? "#define " + u : "",
    t.lightMap ? "#define USE_LIGHTMAP" : "",
    t.aoMap ? "#define USE_AOMAP" : "",
    t.bumpMap ? "#define USE_BUMPMAP" : "",
    t.normalMap ? "#define USE_NORMALMAP" : "",
    t.normalMapObjectSpace ? "#define USE_NORMALMAP_OBJECTSPACE" : "",
    t.normalMapTangentSpace ? "#define USE_NORMALMAP_TANGENTSPACE" : "",
    t.displacementMap ? "#define USE_DISPLACEMENTMAP" : "",
    t.emissiveMap ? "#define USE_EMISSIVEMAP" : "",
    t.anisotropy ? "#define USE_ANISOTROPY" : "",
    t.anisotropyMap ? "#define USE_ANISOTROPYMAP" : "",
    t.clearcoatMap ? "#define USE_CLEARCOATMAP" : "",
    t.clearcoatRoughnessMap ? "#define USE_CLEARCOAT_ROUGHNESSMAP" : "",
    t.clearcoatNormalMap ? "#define USE_CLEARCOAT_NORMALMAP" : "",
    t.iridescenceMap ? "#define USE_IRIDESCENCEMAP" : "",
    t.iridescenceThicknessMap ? "#define USE_IRIDESCENCE_THICKNESSMAP" : "",
    t.specularMap ? "#define USE_SPECULARMAP" : "",
    t.specularColorMap ? "#define USE_SPECULAR_COLORMAP" : "",
    t.specularIntensityMap ? "#define USE_SPECULAR_INTENSITYMAP" : "",
    t.roughnessMap ? "#define USE_ROUGHNESSMAP" : "",
    t.metalnessMap ? "#define USE_METALNESSMAP" : "",
    t.alphaMap ? "#define USE_ALPHAMAP" : "",
    t.alphaHash ? "#define USE_ALPHAHASH" : "",
    t.transmission ? "#define USE_TRANSMISSION" : "",
    t.transmissionMap ? "#define USE_TRANSMISSIONMAP" : "",
    t.thicknessMap ? "#define USE_THICKNESSMAP" : "",
    t.sheenColorMap ? "#define USE_SHEEN_COLORMAP" : "",
    t.sheenRoughnessMap ? "#define USE_SHEEN_ROUGHNESSMAP" : "",
    //
    t.mapUv ? "#define MAP_UV " + t.mapUv : "",
    t.alphaMapUv ? "#define ALPHAMAP_UV " + t.alphaMapUv : "",
    t.lightMapUv ? "#define LIGHTMAP_UV " + t.lightMapUv : "",
    t.aoMapUv ? "#define AOMAP_UV " + t.aoMapUv : "",
    t.emissiveMapUv ? "#define EMISSIVEMAP_UV " + t.emissiveMapUv : "",
    t.bumpMapUv ? "#define BUMPMAP_UV " + t.bumpMapUv : "",
    t.normalMapUv ? "#define NORMALMAP_UV " + t.normalMapUv : "",
    t.displacementMapUv ? "#define DISPLACEMENTMAP_UV " + t.displacementMapUv : "",
    t.metalnessMapUv ? "#define METALNESSMAP_UV " + t.metalnessMapUv : "",
    t.roughnessMapUv ? "#define ROUGHNESSMAP_UV " + t.roughnessMapUv : "",
    t.anisotropyMapUv ? "#define ANISOTROPYMAP_UV " + t.anisotropyMapUv : "",
    t.clearcoatMapUv ? "#define CLEARCOATMAP_UV " + t.clearcoatMapUv : "",
    t.clearcoatNormalMapUv ? "#define CLEARCOAT_NORMALMAP_UV " + t.clearcoatNormalMapUv : "",
    t.clearcoatRoughnessMapUv ? "#define CLEARCOAT_ROUGHNESSMAP_UV " + t.clearcoatRoughnessMapUv : "",
    t.iridescenceMapUv ? "#define IRIDESCENCEMAP_UV " + t.iridescenceMapUv : "",
    t.iridescenceThicknessMapUv ? "#define IRIDESCENCE_THICKNESSMAP_UV " + t.iridescenceThicknessMapUv : "",
    t.sheenColorMapUv ? "#define SHEEN_COLORMAP_UV " + t.sheenColorMapUv : "",
    t.sheenRoughnessMapUv ? "#define SHEEN_ROUGHNESSMAP_UV " + t.sheenRoughnessMapUv : "",
    t.specularMapUv ? "#define SPECULARMAP_UV " + t.specularMapUv : "",
    t.specularColorMapUv ? "#define SPECULAR_COLORMAP_UV " + t.specularColorMapUv : "",
    t.specularIntensityMapUv ? "#define SPECULAR_INTENSITYMAP_UV " + t.specularIntensityMapUv : "",
    t.transmissionMapUv ? "#define TRANSMISSIONMAP_UV " + t.transmissionMapUv : "",
    t.thicknessMapUv ? "#define THICKNESSMAP_UV " + t.thicknessMapUv : "",
    //
    t.vertexTangents && t.flatShading === !1 ? "#define USE_TANGENT" : "",
    t.vertexColors ? "#define USE_COLOR" : "",
    t.vertexAlphas ? "#define USE_COLOR_ALPHA" : "",
    t.vertexUv1s ? "#define USE_UV1" : "",
    t.vertexUv2s ? "#define USE_UV2" : "",
    t.vertexUv3s ? "#define USE_UV3" : "",
    t.pointsUvs ? "#define USE_POINTS_UV" : "",
    t.flatShading ? "#define FLAT_SHADED" : "",
    t.skinning ? "#define USE_SKINNING" : "",
    t.morphTargets ? "#define USE_MORPHTARGETS" : "",
    t.morphNormals && t.flatShading === !1 ? "#define USE_MORPHNORMALS" : "",
    t.morphColors ? "#define USE_MORPHCOLORS" : "",
    t.morphTargetsCount > 0 ? "#define MORPHTARGETS_TEXTURE_STRIDE " + t.morphTextureStride : "",
    t.morphTargetsCount > 0 ? "#define MORPHTARGETS_COUNT " + t.morphTargetsCount : "",
    t.doubleSided ? "#define DOUBLE_SIDED" : "",
    t.flipSided ? "#define FLIP_SIDED" : "",
    t.shadowMapEnabled ? "#define USE_SHADOWMAP" : "",
    t.shadowMapEnabled ? "#define " + l : "",
    t.sizeAttenuation ? "#define USE_SIZEATTENUATION" : "",
    t.numLightProbes > 0 ? "#define USE_LIGHT_PROBES" : "",
    t.logarithmicDepthBuffer ? "#define USE_LOGARITHMIC_DEPTH_BUFFER" : "",
    t.reversedDepthBuffer ? "#define USE_REVERSED_DEPTH_BUFFER" : "",
    "uniform mat4 modelMatrix;",
    "uniform mat4 modelViewMatrix;",
    "uniform mat4 projectionMatrix;",
    "uniform mat4 viewMatrix;",
    "uniform mat3 normalMatrix;",
    "uniform vec3 cameraPosition;",
    "uniform bool isOrthographic;",
    "#ifdef USE_INSTANCING",
    "	attribute mat4 instanceMatrix;",
    "#endif",
    "#ifdef USE_INSTANCING_COLOR",
    "	attribute vec3 instanceColor;",
    "#endif",
    "#ifdef USE_INSTANCING_MORPH",
    "	uniform sampler2D morphTexture;",
    "#endif",
    "attribute vec3 position;",
    "attribute vec3 normal;",
    "attribute vec2 uv;",
    "#ifdef USE_UV1",
    "	attribute vec2 uv1;",
    "#endif",
    "#ifdef USE_UV2",
    "	attribute vec2 uv2;",
    "#endif",
    "#ifdef USE_UV3",
    "	attribute vec2 uv3;",
    "#endif",
    "#ifdef USE_TANGENT",
    "	attribute vec4 tangent;",
    "#endif",
    "#if defined( USE_COLOR_ALPHA )",
    "	attribute vec4 color;",
    "#elif defined( USE_COLOR )",
    "	attribute vec3 color;",
    "#endif",
    "#ifdef USE_SKINNING",
    "	attribute vec4 skinIndex;",
    "	attribute vec4 skinWeight;",
    "#endif",
    `
`
  ].filter(or).join(`
`), d = [
    Ch(t),
    "#define SHADER_TYPE " + t.shaderType,
    "#define SHADER_NAME " + t.shaderName,
    v,
    t.useFog && t.fog ? "#define USE_FOG" : "",
    t.useFog && t.fogExp2 ? "#define FOG_EXP2" : "",
    t.alphaToCoverage ? "#define ALPHA_TO_COVERAGE" : "",
    t.map ? "#define USE_MAP" : "",
    t.matcap ? "#define USE_MATCAP" : "",
    t.envMap ? "#define USE_ENVMAP" : "",
    t.envMap ? "#define " + c : "",
    t.envMap ? "#define " + u : "",
    t.envMap ? "#define " + h : "",
    f ? "#define CUBEUV_TEXEL_WIDTH " + f.texelWidth : "",
    f ? "#define CUBEUV_TEXEL_HEIGHT " + f.texelHeight : "",
    f ? "#define CUBEUV_MAX_MIP " + f.maxMip + ".0" : "",
    t.lightMap ? "#define USE_LIGHTMAP" : "",
    t.aoMap ? "#define USE_AOMAP" : "",
    t.bumpMap ? "#define USE_BUMPMAP" : "",
    t.normalMap ? "#define USE_NORMALMAP" : "",
    t.normalMapObjectSpace ? "#define USE_NORMALMAP_OBJECTSPACE" : "",
    t.normalMapTangentSpace ? "#define USE_NORMALMAP_TANGENTSPACE" : "",
    t.emissiveMap ? "#define USE_EMISSIVEMAP" : "",
    t.anisotropy ? "#define USE_ANISOTROPY" : "",
    t.anisotropyMap ? "#define USE_ANISOTROPYMAP" : "",
    t.clearcoat ? "#define USE_CLEARCOAT" : "",
    t.clearcoatMap ? "#define USE_CLEARCOATMAP" : "",
    t.clearcoatRoughnessMap ? "#define USE_CLEARCOAT_ROUGHNESSMAP" : "",
    t.clearcoatNormalMap ? "#define USE_CLEARCOAT_NORMALMAP" : "",
    t.dispersion ? "#define USE_DISPERSION" : "",
    t.iridescence ? "#define USE_IRIDESCENCE" : "",
    t.iridescenceMap ? "#define USE_IRIDESCENCEMAP" : "",
    t.iridescenceThicknessMap ? "#define USE_IRIDESCENCE_THICKNESSMAP" : "",
    t.specularMap ? "#define USE_SPECULARMAP" : "",
    t.specularColorMap ? "#define USE_SPECULAR_COLORMAP" : "",
    t.specularIntensityMap ? "#define USE_SPECULAR_INTENSITYMAP" : "",
    t.roughnessMap ? "#define USE_ROUGHNESSMAP" : "",
    t.metalnessMap ? "#define USE_METALNESSMAP" : "",
    t.alphaMap ? "#define USE_ALPHAMAP" : "",
    t.alphaTest ? "#define USE_ALPHATEST" : "",
    t.alphaHash ? "#define USE_ALPHAHASH" : "",
    t.sheen ? "#define USE_SHEEN" : "",
    t.sheenColorMap ? "#define USE_SHEEN_COLORMAP" : "",
    t.sheenRoughnessMap ? "#define USE_SHEEN_ROUGHNESSMAP" : "",
    t.transmission ? "#define USE_TRANSMISSION" : "",
    t.transmissionMap ? "#define USE_TRANSMISSIONMAP" : "",
    t.thicknessMap ? "#define USE_THICKNESSMAP" : "",
    t.vertexTangents && t.flatShading === !1 ? "#define USE_TANGENT" : "",
    t.vertexColors || t.instancingColor || t.batchingColor ? "#define USE_COLOR" : "",
    t.vertexAlphas ? "#define USE_COLOR_ALPHA" : "",
    t.vertexUv1s ? "#define USE_UV1" : "",
    t.vertexUv2s ? "#define USE_UV2" : "",
    t.vertexUv3s ? "#define USE_UV3" : "",
    t.pointsUvs ? "#define USE_POINTS_UV" : "",
    t.gradientMap ? "#define USE_GRADIENTMAP" : "",
    t.flatShading ? "#define FLAT_SHADED" : "",
    t.doubleSided ? "#define DOUBLE_SIDED" : "",
    t.flipSided ? "#define FLIP_SIDED" : "",
    t.shadowMapEnabled ? "#define USE_SHADOWMAP" : "",
    t.shadowMapEnabled ? "#define " + l : "",
    t.premultipliedAlpha ? "#define PREMULTIPLIED_ALPHA" : "",
    t.numLightProbes > 0 ? "#define USE_LIGHT_PROBES" : "",
    t.decodeVideoTexture ? "#define DECODE_VIDEO_TEXTURE" : "",
    t.decodeVideoTextureEmissive ? "#define DECODE_VIDEO_TEXTURE_EMISSIVE" : "",
    t.logarithmicDepthBuffer ? "#define USE_LOGARITHMIC_DEPTH_BUFFER" : "",
    t.reversedDepthBuffer ? "#define USE_REVERSED_DEPTH_BUFFER" : "",
    "uniform mat4 viewMatrix;",
    "uniform vec3 cameraPosition;",
    "uniform bool isOrthographic;",
    t.toneMapping !== Mi ? "#define TONE_MAPPING" : "",
    t.toneMapping !== Mi ? je.tonemapping_pars_fragment : "",
    // this code is required here because it is used by the toneMapping() function defined below
    t.toneMapping !== Mi ? mS("toneMapping", t.toneMapping) : "",
    t.dithering ? "#define DITHERING" : "",
    t.opaque ? "#define OPAQUE" : "",
    je.colorspace_pars_fragment,
    // this code is required here because it is used by the various encoding/decoding function defined below
    pS("linearToOutputTexel", t.outputColorSpace),
    _S(),
    t.useDepthPacking ? "#define DEPTH_PACKING " + t.depthPacking : "",
    `
`
  ].filter(or).join(`
`)), o = tc(o), o = Ah(o, t), o = wh(o, t), a = tc(a), a = Ah(a, t), a = wh(a, t), o = Rh(o), a = Rh(a), t.isRawShaderMaterial !== !0 && (b = `#version 300 es
`, m = [
    p,
    "#define attribute in",
    "#define varying out",
    "#define texture2D texture"
  ].join(`
`) + `
` + m, d = [
    "#define varying in",
    t.glslVersion === Lu ? "" : "layout(location = 0) out highp vec4 pc_fragColor;",
    t.glslVersion === Lu ? "" : "#define gl_FragColor pc_fragColor",
    "#define gl_FragDepthEXT gl_FragDepth",
    "#define texture2D texture",
    "#define textureCube texture",
    "#define texture2DProj textureProj",
    "#define texture2DLodEXT textureLod",
    "#define texture2DProjLodEXT textureProjLod",
    "#define textureCubeLodEXT textureLod",
    "#define texture2DGradEXT textureGrad",
    "#define texture2DProjGradEXT textureProjGrad",
    "#define textureCubeGradEXT textureGrad"
  ].join(`
`) + `
` + d);
  const A = b + m + o, M = b + d + a, C = Eh(s, s.VERTEX_SHADER, A), w = Eh(s, s.FRAGMENT_SHADER, M);
  s.attachShader(x, C), s.attachShader(x, w), t.index0AttributeName !== void 0 ? s.bindAttribLocation(x, 0, t.index0AttributeName) : t.morphTargets === !0 && s.bindAttribLocation(x, 0, "position"), s.linkProgram(x);
  function P(D) {
    if (n.debug.checkShaderErrors) {
      const L = s.getProgramInfoLog(x) || "", V = s.getShaderInfoLog(C) || "", Z = s.getShaderInfoLog(w) || "", ne = L.trim(), J = V.trim(), ie = Z.trim();
      let H = !0, fe = !0;
      if (s.getProgramParameter(x, s.LINK_STATUS) === !1)
        if (H = !1, typeof n.debug.onShaderError == "function")
          n.debug.onShaderError(s, x, C, w);
        else {
          const ge = bh(s, C, "vertex"), ye = bh(s, w, "fragment");
          console.error(
            "THREE.WebGLProgram: Shader Error " + s.getError() + " - VALIDATE_STATUS " + s.getProgramParameter(x, s.VALIDATE_STATUS) + `

Material Name: ` + D.name + `
Material Type: ` + D.type + `

Program Info Log: ` + ne + `
` + ge + `
` + ye
          );
        }
      else ne !== "" ? console.warn("THREE.WebGLProgram: Program Info Log:", ne) : (J === "" || ie === "") && (fe = !1);
      fe && (D.diagnostics = {
        runnable: H,
        programLog: ne,
        vertexShader: {
          log: J,
          prefix: m
        },
        fragmentShader: {
          log: ie,
          prefix: d
        }
      });
    }
    s.deleteShader(C), s.deleteShader(w), U = new Ao(s, x), S = xS(s, x);
  }
  let U;
  this.getUniforms = function() {
    return U === void 0 && P(this), U;
  };
  let S;
  this.getAttributes = function() {
    return S === void 0 && P(this), S;
  };
  let y = t.rendererExtensionParallelShaderCompile === !1;
  return this.isReady = function() {
    return y === !1 && (y = s.getProgramParameter(x, uS)), y;
  }, this.destroy = function() {
    i.releaseStatesOfProgram(this), s.deleteProgram(x), this.program = void 0;
  }, this.type = t.shaderType, this.name = t.shaderName, this.id = hS++, this.cacheKey = e, this.usedTimes = 1, this.program = x, this.vertexShader = C, this.fragmentShader = w, this;
}
let DS = 0;
class LS {
  constructor() {
    this.shaderCache = /* @__PURE__ */ new Map(), this.materialCache = /* @__PURE__ */ new Map();
  }
  update(e) {
    const t = e.vertexShader, i = e.fragmentShader, s = this._getShaderStage(t), r = this._getShaderStage(i), o = this._getShaderCacheForMaterial(e);
    return o.has(s) === !1 && (o.add(s), s.usedTimes++), o.has(r) === !1 && (o.add(r), r.usedTimes++), this;
  }
  remove(e) {
    const t = this.materialCache.get(e);
    for (const i of t)
      i.usedTimes--, i.usedTimes === 0 && this.shaderCache.delete(i.code);
    return this.materialCache.delete(e), this;
  }
  getVertexShaderID(e) {
    return this._getShaderStage(e.vertexShader).id;
  }
  getFragmentShaderID(e) {
    return this._getShaderStage(e.fragmentShader).id;
  }
  dispose() {
    this.shaderCache.clear(), this.materialCache.clear();
  }
  _getShaderCacheForMaterial(e) {
    const t = this.materialCache;
    let i = t.get(e);
    return i === void 0 && (i = /* @__PURE__ */ new Set(), t.set(e, i)), i;
  }
  _getShaderStage(e) {
    const t = this.shaderCache;
    let i = t.get(e);
    return i === void 0 && (i = new IS(e), t.set(e, i)), i;
  }
}
class IS {
  constructor(e) {
    this.id = DS++, this.code = e, this.usedTimes = 0;
  }
}
function US(n, e, t, i, s, r, o) {
  const a = new _d(), l = new LS(), c = /* @__PURE__ */ new Set(), u = [], h = s.logarithmicDepthBuffer, f = s.vertexTextures;
  let p = s.precision;
  const v = {
    MeshDepthMaterial: "depth",
    MeshDistanceMaterial: "distanceRGBA",
    MeshNormalMaterial: "normal",
    MeshBasicMaterial: "basic",
    MeshLambertMaterial: "lambert",
    MeshPhongMaterial: "phong",
    MeshToonMaterial: "toon",
    MeshStandardMaterial: "physical",
    MeshPhysicalMaterial: "physical",
    MeshMatcapMaterial: "matcap",
    LineBasicMaterial: "basic",
    LineDashedMaterial: "dashed",
    PointsMaterial: "points",
    ShadowMaterial: "shadow",
    SpriteMaterial: "sprite"
  };
  function x(S) {
    return c.add(S), S === 0 ? "uv" : `uv${S}`;
  }
  function m(S, y, D, L, V) {
    const Z = L.fog, ne = V.geometry, J = S.isMeshStandardMaterial ? L.environment : null, ie = (S.isMeshStandardMaterial ? t : e).get(S.envMap || J), H = ie && ie.mapping === ta ? ie.image.height : null, fe = v[S.type];
    S.precision !== null && (p = s.getMaxPrecision(S.precision), p !== S.precision && console.warn("THREE.WebGLProgram.getParameters:", S.precision, "not supported, using", p, "instead."));
    const ge = ne.morphAttributes.position || ne.morphAttributes.normal || ne.morphAttributes.color, ye = ge !== void 0 ? ge.length : 0;
    let Fe = 0;
    ne.morphAttributes.position !== void 0 && (Fe = 1), ne.morphAttributes.normal !== void 0 && (Fe = 2), ne.morphAttributes.color !== void 0 && (Fe = 3);
    let Je, Ge, Ae, X;
    if (fe) {
      const nt = Ln[fe];
      Je = nt.vertexShader, Ge = nt.fragmentShader;
    } else
      Je = S.vertexShader, Ge = S.fragmentShader, l.update(S), Ae = l.getVertexShaderID(S), X = l.getFragmentShaderID(S);
    const re = n.getRenderTarget(), be = n.state.buffers.depth.getReversed(), Be = V.isInstancedMesh === !0, Pe = V.isBatchedMesh === !0, Ze = !!S.map, R = !!S.matcap, g = !!ie, W = !!S.aoMap, K = !!S.lightMap, Y = !!S.bumpMap, z = !!S.normalMap, ae = !!S.displacementMap, j = !!S.emissiveMap, ee = !!S.metalnessMap, te = !!S.roughnessMap, xe = S.anisotropy > 0, E = S.clearcoat > 0, _ = S.dispersion > 0, I = S.iridescence > 0, k = S.sheen > 0, Q = S.transmission > 0, G = xe && !!S.anisotropyMap, pe = E && !!S.clearcoatMap, oe = E && !!S.clearcoatNormalMap, Se = E && !!S.clearcoatRoughnessMap, Ee = I && !!S.iridescenceMap, le = I && !!S.iridescenceThicknessMap, ve = k && !!S.sheenColorMap, Ce = k && !!S.sheenRoughnessMap, Te = !!S.specularMap, me = !!S.specularColorMap, ke = !!S.specularIntensityMap, F = Q && !!S.transmissionMap, he = Q && !!S.thicknessMap, de = !!S.gradientMap, Re = !!S.alphaMap, ce = S.alphaTest > 0, se = !!S.alphaHash, Le = !!S.extensions;
    let We = Mi;
    S.toneMapped && (re === null || re.isXRRenderTarget === !0) && (We = n.toneMapping);
    const ht = {
      shaderID: fe,
      shaderType: S.type,
      shaderName: S.name,
      vertexShader: Je,
      fragmentShader: Ge,
      defines: S.defines,
      customVertexShaderID: Ae,
      customFragmentShaderID: X,
      isRawShaderMaterial: S.isRawShaderMaterial === !0,
      glslVersion: S.glslVersion,
      precision: p,
      batching: Pe,
      batchingColor: Pe && V._colorsTexture !== null,
      instancing: Be,
      instancingColor: Be && V.instanceColor !== null,
      instancingMorph: Be && V.morphTexture !== null,
      supportsVertexTextures: f,
      outputColorSpace: re === null ? n.outputColorSpace : re.isXRRenderTarget === !0 ? re.texture.colorSpace : Bs,
      alphaToCoverage: !!S.alphaToCoverage,
      map: Ze,
      matcap: R,
      envMap: g,
      envMapMode: g && ie.mapping,
      envMapCubeUVHeight: H,
      aoMap: W,
      lightMap: K,
      bumpMap: Y,
      normalMap: z,
      displacementMap: f && ae,
      emissiveMap: j,
      normalMapObjectSpace: z && S.normalMapType === Eg,
      normalMapTangentSpace: z && S.normalMapType === fd,
      metalnessMap: ee,
      roughnessMap: te,
      anisotropy: xe,
      anisotropyMap: G,
      clearcoat: E,
      clearcoatMap: pe,
      clearcoatNormalMap: oe,
      clearcoatRoughnessMap: Se,
      dispersion: _,
      iridescence: I,
      iridescenceMap: Ee,
      iridescenceThicknessMap: le,
      sheen: k,
      sheenColorMap: ve,
      sheenRoughnessMap: Ce,
      specularMap: Te,
      specularColorMap: me,
      specularIntensityMap: ke,
      transmission: Q,
      transmissionMap: F,
      thicknessMap: he,
      gradientMap: de,
      opaque: S.transparent === !1 && S.blending === Ls && S.alphaToCoverage === !1,
      alphaMap: Re,
      alphaTest: ce,
      alphaHash: se,
      combine: S.combine,
      //
      mapUv: Ze && x(S.map.channel),
      aoMapUv: W && x(S.aoMap.channel),
      lightMapUv: K && x(S.lightMap.channel),
      bumpMapUv: Y && x(S.bumpMap.channel),
      normalMapUv: z && x(S.normalMap.channel),
      displacementMapUv: ae && x(S.displacementMap.channel),
      emissiveMapUv: j && x(S.emissiveMap.channel),
      metalnessMapUv: ee && x(S.metalnessMap.channel),
      roughnessMapUv: te && x(S.roughnessMap.channel),
      anisotropyMapUv: G && x(S.anisotropyMap.channel),
      clearcoatMapUv: pe && x(S.clearcoatMap.channel),
      clearcoatNormalMapUv: oe && x(S.clearcoatNormalMap.channel),
      clearcoatRoughnessMapUv: Se && x(S.clearcoatRoughnessMap.channel),
      iridescenceMapUv: Ee && x(S.iridescenceMap.channel),
      iridescenceThicknessMapUv: le && x(S.iridescenceThicknessMap.channel),
      sheenColorMapUv: ve && x(S.sheenColorMap.channel),
      sheenRoughnessMapUv: Ce && x(S.sheenRoughnessMap.channel),
      specularMapUv: Te && x(S.specularMap.channel),
      specularColorMapUv: me && x(S.specularColorMap.channel),
      specularIntensityMapUv: ke && x(S.specularIntensityMap.channel),
      transmissionMapUv: F && x(S.transmissionMap.channel),
      thicknessMapUv: he && x(S.thicknessMap.channel),
      alphaMapUv: Re && x(S.alphaMap.channel),
      //
      vertexTangents: !!ne.attributes.tangent && (z || xe),
      vertexColors: S.vertexColors,
      vertexAlphas: S.vertexColors === !0 && !!ne.attributes.color && ne.attributes.color.itemSize === 4,
      pointsUvs: V.isPoints === !0 && !!ne.attributes.uv && (Ze || Re),
      fog: !!Z,
      useFog: S.fog === !0,
      fogExp2: !!Z && Z.isFogExp2,
      flatShading: S.flatShading === !0 && S.wireframe === !1,
      sizeAttenuation: S.sizeAttenuation === !0,
      logarithmicDepthBuffer: h,
      reversedDepthBuffer: be,
      skinning: V.isSkinnedMesh === !0,
      morphTargets: ne.morphAttributes.position !== void 0,
      morphNormals: ne.morphAttributes.normal !== void 0,
      morphColors: ne.morphAttributes.color !== void 0,
      morphTargetsCount: ye,
      morphTextureStride: Fe,
      numDirLights: y.directional.length,
      numPointLights: y.point.length,
      numSpotLights: y.spot.length,
      numSpotLightMaps: y.spotLightMap.length,
      numRectAreaLights: y.rectArea.length,
      numHemiLights: y.hemi.length,
      numDirLightShadows: y.directionalShadowMap.length,
      numPointLightShadows: y.pointShadowMap.length,
      numSpotLightShadows: y.spotShadowMap.length,
      numSpotLightShadowsWithMaps: y.numSpotLightShadowsWithMaps,
      numLightProbes: y.numLightProbes,
      numClippingPlanes: o.numPlanes,
      numClipIntersection: o.numIntersection,
      dithering: S.dithering,
      shadowMapEnabled: n.shadowMap.enabled && D.length > 0,
      shadowMapType: n.shadowMap.type,
      toneMapping: We,
      decodeVideoTexture: Ze && S.map.isVideoTexture === !0 && et.getTransfer(S.map.colorSpace) === ot,
      decodeVideoTextureEmissive: j && S.emissiveMap.isVideoTexture === !0 && et.getTransfer(S.emissiveMap.colorSpace) === ot,
      premultipliedAlpha: S.premultipliedAlpha,
      doubleSided: S.side === Qn,
      flipSided: S.side === Wt,
      useDepthPacking: S.depthPacking >= 0,
      depthPacking: S.depthPacking || 0,
      index0AttributeName: S.index0AttributeName,
      extensionClipCullDistance: Le && S.extensions.clipCullDistance === !0 && i.has("WEBGL_clip_cull_distance"),
      extensionMultiDraw: (Le && S.extensions.multiDraw === !0 || Pe) && i.has("WEBGL_multi_draw"),
      rendererExtensionParallelShaderCompile: i.has("KHR_parallel_shader_compile"),
      customProgramCacheKey: S.customProgramCacheKey()
    };
    return ht.vertexUv1s = c.has(1), ht.vertexUv2s = c.has(2), ht.vertexUv3s = c.has(3), c.clear(), ht;
  }
  function d(S) {
    const y = [];
    if (S.shaderID ? y.push(S.shaderID) : (y.push(S.customVertexShaderID), y.push(S.customFragmentShaderID)), S.defines !== void 0)
      for (const D in S.defines)
        y.push(D), y.push(S.defines[D]);
    return S.isRawShaderMaterial === !1 && (b(y, S), A(y, S), y.push(n.outputColorSpace)), y.push(S.customProgramCacheKey), y.join();
  }
  function b(S, y) {
    S.push(y.precision), S.push(y.outputColorSpace), S.push(y.envMapMode), S.push(y.envMapCubeUVHeight), S.push(y.mapUv), S.push(y.alphaMapUv), S.push(y.lightMapUv), S.push(y.aoMapUv), S.push(y.bumpMapUv), S.push(y.normalMapUv), S.push(y.displacementMapUv), S.push(y.emissiveMapUv), S.push(y.metalnessMapUv), S.push(y.roughnessMapUv), S.push(y.anisotropyMapUv), S.push(y.clearcoatMapUv), S.push(y.clearcoatNormalMapUv), S.push(y.clearcoatRoughnessMapUv), S.push(y.iridescenceMapUv), S.push(y.iridescenceThicknessMapUv), S.push(y.sheenColorMapUv), S.push(y.sheenRoughnessMapUv), S.push(y.specularMapUv), S.push(y.specularColorMapUv), S.push(y.specularIntensityMapUv), S.push(y.transmissionMapUv), S.push(y.thicknessMapUv), S.push(y.combine), S.push(y.fogExp2), S.push(y.sizeAttenuation), S.push(y.morphTargetsCount), S.push(y.morphAttributeCount), S.push(y.numDirLights), S.push(y.numPointLights), S.push(y.numSpotLights), S.push(y.numSpotLightMaps), S.push(y.numHemiLights), S.push(y.numRectAreaLights), S.push(y.numDirLightShadows), S.push(y.numPointLightShadows), S.push(y.numSpotLightShadows), S.push(y.numSpotLightShadowsWithMaps), S.push(y.numLightProbes), S.push(y.shadowMapType), S.push(y.toneMapping), S.push(y.numClippingPlanes), S.push(y.numClipIntersection), S.push(y.depthPacking);
  }
  function A(S, y) {
    a.disableAll(), y.supportsVertexTextures && a.enable(0), y.instancing && a.enable(1), y.instancingColor && a.enable(2), y.instancingMorph && a.enable(3), y.matcap && a.enable(4), y.envMap && a.enable(5), y.normalMapObjectSpace && a.enable(6), y.normalMapTangentSpace && a.enable(7), y.clearcoat && a.enable(8), y.iridescence && a.enable(9), y.alphaTest && a.enable(10), y.vertexColors && a.enable(11), y.vertexAlphas && a.enable(12), y.vertexUv1s && a.enable(13), y.vertexUv2s && a.enable(14), y.vertexUv3s && a.enable(15), y.vertexTangents && a.enable(16), y.anisotropy && a.enable(17), y.alphaHash && a.enable(18), y.batching && a.enable(19), y.dispersion && a.enable(20), y.batchingColor && a.enable(21), y.gradientMap && a.enable(22), S.push(a.mask), a.disableAll(), y.fog && a.enable(0), y.useFog && a.enable(1), y.flatShading && a.enable(2), y.logarithmicDepthBuffer && a.enable(3), y.reversedDepthBuffer && a.enable(4), y.skinning && a.enable(5), y.morphTargets && a.enable(6), y.morphNormals && a.enable(7), y.morphColors && a.enable(8), y.premultipliedAlpha && a.enable(9), y.shadowMapEnabled && a.enable(10), y.doubleSided && a.enable(11), y.flipSided && a.enable(12), y.useDepthPacking && a.enable(13), y.dithering && a.enable(14), y.transmission && a.enable(15), y.sheen && a.enable(16), y.opaque && a.enable(17), y.pointsUvs && a.enable(18), y.decodeVideoTexture && a.enable(19), y.decodeVideoTextureEmissive && a.enable(20), y.alphaToCoverage && a.enable(21), S.push(a.mask);
  }
  function M(S) {
    const y = v[S.type];
    let D;
    if (y) {
      const L = Ln[y];
      D = Qg.clone(L.uniforms);
    } else
      D = S.uniforms;
    return D;
  }
  function C(S, y) {
    let D;
    for (let L = 0, V = u.length; L < V; L++) {
      const Z = u[L];
      if (Z.cacheKey === y) {
        D = Z, ++D.usedTimes;
        break;
      }
    }
    return D === void 0 && (D = new PS(n, y, S, r), u.push(D)), D;
  }
  function w(S) {
    if (--S.usedTimes === 0) {
      const y = u.indexOf(S);
      u[y] = u[u.length - 1], u.pop(), S.destroy();
    }
  }
  function P(S) {
    l.remove(S);
  }
  function U() {
    l.dispose();
  }
  return {
    getParameters: m,
    getProgramCacheKey: d,
    getUniforms: M,
    acquireProgram: C,
    releaseProgram: w,
    releaseShaderCache: P,
    // Exposed for resource monitoring & error feedback via renderer.info:
    programs: u,
    dispose: U
  };
}
function NS() {
  let n = /* @__PURE__ */ new WeakMap();
  function e(o) {
    return n.has(o);
  }
  function t(o) {
    let a = n.get(o);
    return a === void 0 && (a = {}, n.set(o, a)), a;
  }
  function i(o) {
    n.delete(o);
  }
  function s(o, a, l) {
    n.get(o)[a] = l;
  }
  function r() {
    n = /* @__PURE__ */ new WeakMap();
  }
  return {
    has: e,
    get: t,
    remove: i,
    update: s,
    dispose: r
  };
}
function FS(n, e) {
  return n.groupOrder !== e.groupOrder ? n.groupOrder - e.groupOrder : n.renderOrder !== e.renderOrder ? n.renderOrder - e.renderOrder : n.material.id !== e.material.id ? n.material.id - e.material.id : n.z !== e.z ? n.z - e.z : n.id - e.id;
}
function Ph(n, e) {
  return n.groupOrder !== e.groupOrder ? n.groupOrder - e.groupOrder : n.renderOrder !== e.renderOrder ? n.renderOrder - e.renderOrder : n.z !== e.z ? e.z - n.z : n.id - e.id;
}
function Dh() {
  const n = [];
  let e = 0;
  const t = [], i = [], s = [];
  function r() {
    e = 0, t.length = 0, i.length = 0, s.length = 0;
  }
  function o(h, f, p, v, x, m) {
    let d = n[e];
    return d === void 0 ? (d = {
      id: h.id,
      object: h,
      geometry: f,
      material: p,
      groupOrder: v,
      renderOrder: h.renderOrder,
      z: x,
      group: m
    }, n[e] = d) : (d.id = h.id, d.object = h, d.geometry = f, d.material = p, d.groupOrder = v, d.renderOrder = h.renderOrder, d.z = x, d.group = m), e++, d;
  }
  function a(h, f, p, v, x, m) {
    const d = o(h, f, p, v, x, m);
    p.transmission > 0 ? i.push(d) : p.transparent === !0 ? s.push(d) : t.push(d);
  }
  function l(h, f, p, v, x, m) {
    const d = o(h, f, p, v, x, m);
    p.transmission > 0 ? i.unshift(d) : p.transparent === !0 ? s.unshift(d) : t.unshift(d);
  }
  function c(h, f) {
    t.length > 1 && t.sort(h || FS), i.length > 1 && i.sort(f || Ph), s.length > 1 && s.sort(f || Ph);
  }
  function u() {
    for (let h = e, f = n.length; h < f; h++) {
      const p = n[h];
      if (p.id === null) break;
      p.id = null, p.object = null, p.geometry = null, p.material = null, p.group = null;
    }
  }
  return {
    opaque: t,
    transmissive: i,
    transparent: s,
    init: r,
    push: a,
    unshift: l,
    finish: u,
    sort: c
  };
}
function OS() {
  let n = /* @__PURE__ */ new WeakMap();
  function e(i, s) {
    const r = n.get(i);
    let o;
    return r === void 0 ? (o = new Dh(), n.set(i, [o])) : s >= r.length ? (o = new Dh(), r.push(o)) : o = r[s], o;
  }
  function t() {
    n = /* @__PURE__ */ new WeakMap();
  }
  return {
    get: e,
    dispose: t
  };
}
function BS() {
  const n = {};
  return {
    get: function(e) {
      if (n[e.id] !== void 0)
        return n[e.id];
      let t;
      switch (e.type) {
        case "DirectionalLight":
          t = {
            direction: new N(),
            color: new Xe()
          };
          break;
        case "SpotLight":
          t = {
            position: new N(),
            direction: new N(),
            color: new Xe(),
            distance: 0,
            coneCos: 0,
            penumbraCos: 0,
            decay: 0
          };
          break;
        case "PointLight":
          t = {
            position: new N(),
            color: new Xe(),
            distance: 0,
            decay: 0
          };
          break;
        case "HemisphereLight":
          t = {
            direction: new N(),
            skyColor: new Xe(),
            groundColor: new Xe()
          };
          break;
        case "RectAreaLight":
          t = {
            color: new Xe(),
            position: new N(),
            halfWidth: new N(),
            halfHeight: new N()
          };
          break;
      }
      return n[e.id] = t, t;
    }
  };
}
function zS() {
  const n = {};
  return {
    get: function(e) {
      if (n[e.id] !== void 0)
        return n[e.id];
      let t;
      switch (e.type) {
        case "DirectionalLight":
          t = {
            shadowIntensity: 1,
            shadowBias: 0,
            shadowNormalBias: 0,
            shadowRadius: 1,
            shadowMapSize: new Ve()
          };
          break;
        case "SpotLight":
          t = {
            shadowIntensity: 1,
            shadowBias: 0,
            shadowNormalBias: 0,
            shadowRadius: 1,
            shadowMapSize: new Ve()
          };
          break;
        case "PointLight":
          t = {
            shadowIntensity: 1,
            shadowBias: 0,
            shadowNormalBias: 0,
            shadowRadius: 1,
            shadowMapSize: new Ve(),
            shadowCameraNear: 1,
            shadowCameraFar: 1e3
          };
          break;
      }
      return n[e.id] = t, t;
    }
  };
}
let HS = 0;
function VS(n, e) {
  return (e.castShadow ? 2 : 0) - (n.castShadow ? 2 : 0) + (e.map ? 1 : 0) - (n.map ? 1 : 0);
}
function kS(n) {
  const e = new BS(), t = zS(), i = {
    version: 0,
    hash: {
      directionalLength: -1,
      pointLength: -1,
      spotLength: -1,
      rectAreaLength: -1,
      hemiLength: -1,
      numDirectionalShadows: -1,
      numPointShadows: -1,
      numSpotShadows: -1,
      numSpotMaps: -1,
      numLightProbes: -1
    },
    ambient: [0, 0, 0],
    probe: [],
    directional: [],
    directionalShadow: [],
    directionalShadowMap: [],
    directionalShadowMatrix: [],
    spot: [],
    spotLightMap: [],
    spotShadow: [],
    spotShadowMap: [],
    spotLightMatrix: [],
    rectArea: [],
    rectAreaLTC1: null,
    rectAreaLTC2: null,
    point: [],
    pointShadow: [],
    pointShadowMap: [],
    pointShadowMatrix: [],
    hemi: [],
    numSpotLightShadowsWithMaps: 0,
    numLightProbes: 0
  };
  for (let c = 0; c < 9; c++) i.probe.push(new N());
  const s = new N(), r = new pt(), o = new pt();
  function a(c) {
    let u = 0, h = 0, f = 0;
    for (let S = 0; S < 9; S++) i.probe[S].set(0, 0, 0);
    let p = 0, v = 0, x = 0, m = 0, d = 0, b = 0, A = 0, M = 0, C = 0, w = 0, P = 0;
    c.sort(VS);
    for (let S = 0, y = c.length; S < y; S++) {
      const D = c[S], L = D.color, V = D.intensity, Z = D.distance, ne = D.shadow && D.shadow.map ? D.shadow.map.texture : null;
      if (D.isAmbientLight)
        u += L.r * V, h += L.g * V, f += L.b * V;
      else if (D.isLightProbe) {
        for (let J = 0; J < 9; J++)
          i.probe[J].addScaledVector(D.sh.coefficients[J], V);
        P++;
      } else if (D.isDirectionalLight) {
        const J = e.get(D);
        if (J.color.copy(D.color).multiplyScalar(D.intensity), D.castShadow) {
          const ie = D.shadow, H = t.get(D);
          H.shadowIntensity = ie.intensity, H.shadowBias = ie.bias, H.shadowNormalBias = ie.normalBias, H.shadowRadius = ie.radius, H.shadowMapSize = ie.mapSize, i.directionalShadow[p] = H, i.directionalShadowMap[p] = ne, i.directionalShadowMatrix[p] = D.shadow.matrix, b++;
        }
        i.directional[p] = J, p++;
      } else if (D.isSpotLight) {
        const J = e.get(D);
        J.position.setFromMatrixPosition(D.matrixWorld), J.color.copy(L).multiplyScalar(V), J.distance = Z, J.coneCos = Math.cos(D.angle), J.penumbraCos = Math.cos(D.angle * (1 - D.penumbra)), J.decay = D.decay, i.spot[x] = J;
        const ie = D.shadow;
        if (D.map && (i.spotLightMap[C] = D.map, C++, ie.updateMatrices(D), D.castShadow && w++), i.spotLightMatrix[x] = ie.matrix, D.castShadow) {
          const H = t.get(D);
          H.shadowIntensity = ie.intensity, H.shadowBias = ie.bias, H.shadowNormalBias = ie.normalBias, H.shadowRadius = ie.radius, H.shadowMapSize = ie.mapSize, i.spotShadow[x] = H, i.spotShadowMap[x] = ne, M++;
        }
        x++;
      } else if (D.isRectAreaLight) {
        const J = e.get(D);
        J.color.copy(L).multiplyScalar(V), J.halfWidth.set(D.width * 0.5, 0, 0), J.halfHeight.set(0, D.height * 0.5, 0), i.rectArea[m] = J, m++;
      } else if (D.isPointLight) {
        const J = e.get(D);
        if (J.color.copy(D.color).multiplyScalar(D.intensity), J.distance = D.distance, J.decay = D.decay, D.castShadow) {
          const ie = D.shadow, H = t.get(D);
          H.shadowIntensity = ie.intensity, H.shadowBias = ie.bias, H.shadowNormalBias = ie.normalBias, H.shadowRadius = ie.radius, H.shadowMapSize = ie.mapSize, H.shadowCameraNear = ie.camera.near, H.shadowCameraFar = ie.camera.far, i.pointShadow[v] = H, i.pointShadowMap[v] = ne, i.pointShadowMatrix[v] = D.shadow.matrix, A++;
        }
        i.point[v] = J, v++;
      } else if (D.isHemisphereLight) {
        const J = e.get(D);
        J.skyColor.copy(D.color).multiplyScalar(V), J.groundColor.copy(D.groundColor).multiplyScalar(V), i.hemi[d] = J, d++;
      }
    }
    m > 0 && (n.has("OES_texture_float_linear") === !0 ? (i.rectAreaLTC1 = _e.LTC_FLOAT_1, i.rectAreaLTC2 = _e.LTC_FLOAT_2) : (i.rectAreaLTC1 = _e.LTC_HALF_1, i.rectAreaLTC2 = _e.LTC_HALF_2)), i.ambient[0] = u, i.ambient[1] = h, i.ambient[2] = f;
    const U = i.hash;
    (U.directionalLength !== p || U.pointLength !== v || U.spotLength !== x || U.rectAreaLength !== m || U.hemiLength !== d || U.numDirectionalShadows !== b || U.numPointShadows !== A || U.numSpotShadows !== M || U.numSpotMaps !== C || U.numLightProbes !== P) && (i.directional.length = p, i.spot.length = x, i.rectArea.length = m, i.point.length = v, i.hemi.length = d, i.directionalShadow.length = b, i.directionalShadowMap.length = b, i.pointShadow.length = A, i.pointShadowMap.length = A, i.spotShadow.length = M, i.spotShadowMap.length = M, i.directionalShadowMatrix.length = b, i.pointShadowMatrix.length = A, i.spotLightMatrix.length = M + C - w, i.spotLightMap.length = C, i.numSpotLightShadowsWithMaps = w, i.numLightProbes = P, U.directionalLength = p, U.pointLength = v, U.spotLength = x, U.rectAreaLength = m, U.hemiLength = d, U.numDirectionalShadows = b, U.numPointShadows = A, U.numSpotShadows = M, U.numSpotMaps = C, U.numLightProbes = P, i.version = HS++);
  }
  function l(c, u) {
    let h = 0, f = 0, p = 0, v = 0, x = 0;
    const m = u.matrixWorldInverse;
    for (let d = 0, b = c.length; d < b; d++) {
      const A = c[d];
      if (A.isDirectionalLight) {
        const M = i.directional[h];
        M.direction.setFromMatrixPosition(A.matrixWorld), s.setFromMatrixPosition(A.target.matrixWorld), M.direction.sub(s), M.direction.transformDirection(m), h++;
      } else if (A.isSpotLight) {
        const M = i.spot[p];
        M.position.setFromMatrixPosition(A.matrixWorld), M.position.applyMatrix4(m), M.direction.setFromMatrixPosition(A.matrixWorld), s.setFromMatrixPosition(A.target.matrixWorld), M.direction.sub(s), M.direction.transformDirection(m), p++;
      } else if (A.isRectAreaLight) {
        const M = i.rectArea[v];
        M.position.setFromMatrixPosition(A.matrixWorld), M.position.applyMatrix4(m), o.identity(), r.copy(A.matrixWorld), r.premultiply(m), o.extractRotation(r), M.halfWidth.set(A.width * 0.5, 0, 0), M.halfHeight.set(0, A.height * 0.5, 0), M.halfWidth.applyMatrix4(o), M.halfHeight.applyMatrix4(o), v++;
      } else if (A.isPointLight) {
        const M = i.point[f];
        M.position.setFromMatrixPosition(A.matrixWorld), M.position.applyMatrix4(m), f++;
      } else if (A.isHemisphereLight) {
        const M = i.hemi[x];
        M.direction.setFromMatrixPosition(A.matrixWorld), M.direction.transformDirection(m), x++;
      }
    }
  }
  return {
    setup: a,
    setupView: l,
    state: i
  };
}
function Lh(n) {
  const e = new kS(n), t = [], i = [];
  function s(u) {
    c.camera = u, t.length = 0, i.length = 0;
  }
  function r(u) {
    t.push(u);
  }
  function o(u) {
    i.push(u);
  }
  function a() {
    e.setup(t);
  }
  function l(u) {
    e.setupView(t, u);
  }
  const c = {
    lightsArray: t,
    shadowsArray: i,
    camera: null,
    lights: e,
    transmissionRenderTarget: {}
  };
  return {
    init: s,
    state: c,
    setupLights: a,
    setupLightsView: l,
    pushLight: r,
    pushShadow: o
  };
}
function GS(n) {
  let e = /* @__PURE__ */ new WeakMap();
  function t(s, r = 0) {
    const o = e.get(s);
    let a;
    return o === void 0 ? (a = new Lh(n), e.set(s, [a])) : r >= o.length ? (a = new Lh(n), o.push(a)) : a = o[r], a;
  }
  function i() {
    e = /* @__PURE__ */ new WeakMap();
  }
  return {
    get: t,
    dispose: i
  };
}
const WS = `void main() {
	gl_Position = vec4( position, 1.0 );
}`, XS = `uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;
function YS(n, e, t) {
  let i = new wc();
  const s = new Ve(), r = new Ve(), o = new lt(), a = new f0({ depthPacking: yg }), l = new d0(), c = {}, u = t.maxTextureSize, h = { [yi]: Wt, [Wt]: yi, [Qn]: Qn }, f = new Ei({
    defines: {
      VSM_SAMPLES: 8
    },
    uniforms: {
      shadow_pass: { value: null },
      resolution: { value: new Ve() },
      radius: { value: 4 }
    },
    vertexShader: WS,
    fragmentShader: XS
  }), p = f.clone();
  p.defines.HORIZONTAL_PASS = 1;
  const v = new Nt();
  v.setAttribute(
    "position",
    new En(
      new Float32Array([-1, -1, 0.5, 3, -1, 0.5, -1, 3, 0.5]),
      3
    )
  );
  const x = new vt(v, f), m = this;
  this.enabled = !1, this.autoUpdate = !0, this.needsUpdate = !1, this.type = ed;
  let d = this.type;
  this.render = function(w, P, U) {
    if (m.enabled === !1 || m.autoUpdate === !1 && m.needsUpdate === !1 || w.length === 0) return;
    const S = n.getRenderTarget(), y = n.getActiveCubeFace(), D = n.getActiveMipmapLevel(), L = n.state;
    L.setBlending(xi), L.buffers.depth.getReversed() === !0 ? L.buffers.color.setClear(0, 0, 0, 0) : L.buffers.color.setClear(1, 1, 1, 1), L.buffers.depth.setTest(!0), L.setScissorTest(!1);
    const V = d !== jn && this.type === jn, Z = d === jn && this.type !== jn;
    for (let ne = 0, J = w.length; ne < J; ne++) {
      const ie = w[ne], H = ie.shadow;
      if (H === void 0) {
        console.warn("THREE.WebGLShadowMap:", ie, "has no shadow.");
        continue;
      }
      if (H.autoUpdate === !1 && H.needsUpdate === !1) continue;
      s.copy(H.mapSize);
      const fe = H.getFrameExtents();
      if (s.multiply(fe), r.copy(H.mapSize), (s.x > u || s.y > u) && (s.x > u && (r.x = Math.floor(u / fe.x), s.x = r.x * fe.x, H.mapSize.x = r.x), s.y > u && (r.y = Math.floor(u / fe.y), s.y = r.y * fe.y, H.mapSize.y = r.y)), H.map === null || V === !0 || Z === !0) {
        const ye = this.type !== jn ? { minFilter: yn, magFilter: yn } : {};
        H.map !== null && H.map.dispose(), H.map = new ji(s.x, s.y, ye), H.map.texture.name = ie.name + ".shadowMap", H.camera.updateProjectionMatrix();
      }
      n.setRenderTarget(H.map), n.clear();
      const ge = H.getViewportCount();
      for (let ye = 0; ye < ge; ye++) {
        const Fe = H.getViewport(ye);
        o.set(
          r.x * Fe.x,
          r.y * Fe.y,
          r.x * Fe.z,
          r.y * Fe.w
        ), L.viewport(o), H.updateMatrices(ie, ye), i = H.getFrustum(), M(P, U, H.camera, ie, this.type);
      }
      H.isPointLightShadow !== !0 && this.type === jn && b(H, U), H.needsUpdate = !1;
    }
    d = this.type, m.needsUpdate = !1, n.setRenderTarget(S, y, D);
  };
  function b(w, P) {
    const U = e.update(x);
    f.defines.VSM_SAMPLES !== w.blurSamples && (f.defines.VSM_SAMPLES = w.blurSamples, p.defines.VSM_SAMPLES = w.blurSamples, f.needsUpdate = !0, p.needsUpdate = !0), w.mapPass === null && (w.mapPass = new ji(s.x, s.y)), f.uniforms.shadow_pass.value = w.map.texture, f.uniforms.resolution.value = w.mapSize, f.uniforms.radius.value = w.radius, n.setRenderTarget(w.mapPass), n.clear(), n.renderBufferDirect(P, null, U, f, x, null), p.uniforms.shadow_pass.value = w.mapPass.texture, p.uniforms.resolution.value = w.mapSize, p.uniforms.radius.value = w.radius, n.setRenderTarget(w.map), n.clear(), n.renderBufferDirect(P, null, U, p, x, null);
  }
  function A(w, P, U, S) {
    let y = null;
    const D = U.isPointLight === !0 ? w.customDistanceMaterial : w.customDepthMaterial;
    if (D !== void 0)
      y = D;
    else if (y = U.isPointLight === !0 ? l : a, n.localClippingEnabled && P.clipShadows === !0 && Array.isArray(P.clippingPlanes) && P.clippingPlanes.length !== 0 || P.displacementMap && P.displacementScale !== 0 || P.alphaMap && P.alphaTest > 0 || P.map && P.alphaTest > 0 || P.alphaToCoverage === !0) {
      const L = y.uuid, V = P.uuid;
      let Z = c[L];
      Z === void 0 && (Z = {}, c[L] = Z);
      let ne = Z[V];
      ne === void 0 && (ne = y.clone(), Z[V] = ne, P.addEventListener("dispose", C)), y = ne;
    }
    if (y.visible = P.visible, y.wireframe = P.wireframe, S === jn ? y.side = P.shadowSide !== null ? P.shadowSide : P.side : y.side = P.shadowSide !== null ? P.shadowSide : h[P.side], y.alphaMap = P.alphaMap, y.alphaTest = P.alphaToCoverage === !0 ? 0.5 : P.alphaTest, y.map = P.map, y.clipShadows = P.clipShadows, y.clippingPlanes = P.clippingPlanes, y.clipIntersection = P.clipIntersection, y.displacementMap = P.displacementMap, y.displacementScale = P.displacementScale, y.displacementBias = P.displacementBias, y.wireframeLinewidth = P.wireframeLinewidth, y.linewidth = P.linewidth, U.isPointLight === !0 && y.isMeshDistanceMaterial === !0) {
      const L = n.properties.get(y);
      L.light = U;
    }
    return y;
  }
  function M(w, P, U, S, y) {
    if (w.visible === !1) return;
    if (w.layers.test(P.layers) && (w.isMesh || w.isLine || w.isPoints) && (w.castShadow || w.receiveShadow && y === jn) && (!w.frustumCulled || i.intersectsObject(w))) {
      w.modelViewMatrix.multiplyMatrices(U.matrixWorldInverse, w.matrixWorld);
      const V = e.update(w), Z = w.material;
      if (Array.isArray(Z)) {
        const ne = V.groups;
        for (let J = 0, ie = ne.length; J < ie; J++) {
          const H = ne[J], fe = Z[H.materialIndex];
          if (fe && fe.visible) {
            const ge = A(w, fe, S, y);
            w.onBeforeShadow(n, w, P, U, V, ge, H), n.renderBufferDirect(U, null, V, ge, w, H), w.onAfterShadow(n, w, P, U, V, ge, H);
          }
        }
      } else if (Z.visible) {
        const ne = A(w, Z, S, y);
        w.onBeforeShadow(n, w, P, U, V, ne, null), n.renderBufferDirect(U, null, V, ne, w, null), w.onAfterShadow(n, w, P, U, V, ne, null);
      }
    }
    const L = w.children;
    for (let V = 0, Z = L.length; V < Z; V++)
      M(L[V], P, U, S, y);
  }
  function C(w) {
    w.target.removeEventListener("dispose", C);
    for (const U in c) {
      const S = c[U], y = w.target.uuid;
      y in S && (S[y].dispose(), delete S[y]);
    }
  }
}
const qS = {
  [pl]: ml,
  [_l]: xl,
  [gl]: Ml,
  [Ns]: vl,
  [ml]: pl,
  [xl]: _l,
  [Ml]: gl,
  [vl]: Ns
};
function jS(n, e) {
  function t() {
    let F = !1;
    const he = new lt();
    let de = null;
    const Re = new lt(0, 0, 0, 0);
    return {
      setMask: function(ce) {
        de !== ce && !F && (n.colorMask(ce, ce, ce, ce), de = ce);
      },
      setLocked: function(ce) {
        F = ce;
      },
      setClear: function(ce, se, Le, We, ht) {
        ht === !0 && (ce *= We, se *= We, Le *= We), he.set(ce, se, Le, We), Re.equals(he) === !1 && (n.clearColor(ce, se, Le, We), Re.copy(he));
      },
      reset: function() {
        F = !1, de = null, Re.set(-1, 0, 0, 0);
      }
    };
  }
  function i() {
    let F = !1, he = !1, de = null, Re = null, ce = null;
    return {
      setReversed: function(se) {
        if (he !== se) {
          const Le = e.get("EXT_clip_control");
          se ? Le.clipControlEXT(Le.LOWER_LEFT_EXT, Le.ZERO_TO_ONE_EXT) : Le.clipControlEXT(Le.LOWER_LEFT_EXT, Le.NEGATIVE_ONE_TO_ONE_EXT), he = se;
          const We = ce;
          ce = null, this.setClear(We);
        }
      },
      getReversed: function() {
        return he;
      },
      setTest: function(se) {
        se ? re(n.DEPTH_TEST) : be(n.DEPTH_TEST);
      },
      setMask: function(se) {
        de !== se && !F && (n.depthMask(se), de = se);
      },
      setFunc: function(se) {
        if (he && (se = qS[se]), Re !== se) {
          switch (se) {
            case pl:
              n.depthFunc(n.NEVER);
              break;
            case ml:
              n.depthFunc(n.ALWAYS);
              break;
            case _l:
              n.depthFunc(n.LESS);
              break;
            case Ns:
              n.depthFunc(n.LEQUAL);
              break;
            case gl:
              n.depthFunc(n.EQUAL);
              break;
            case vl:
              n.depthFunc(n.GEQUAL);
              break;
            case xl:
              n.depthFunc(n.GREATER);
              break;
            case Ml:
              n.depthFunc(n.NOTEQUAL);
              break;
            default:
              n.depthFunc(n.LEQUAL);
          }
          Re = se;
        }
      },
      setLocked: function(se) {
        F = se;
      },
      setClear: function(se) {
        ce !== se && (he && (se = 1 - se), n.clearDepth(se), ce = se);
      },
      reset: function() {
        F = !1, de = null, Re = null, ce = null, he = !1;
      }
    };
  }
  function s() {
    let F = !1, he = null, de = null, Re = null, ce = null, se = null, Le = null, We = null, ht = null;
    return {
      setTest: function(nt) {
        F || (nt ? re(n.STENCIL_TEST) : be(n.STENCIL_TEST));
      },
      setMask: function(nt) {
        he !== nt && !F && (n.stencilMask(nt), he = nt);
      },
      setFunc: function(nt, Hn, bn) {
        (de !== nt || Re !== Hn || ce !== bn) && (n.stencilFunc(nt, Hn, bn), de = nt, Re = Hn, ce = bn);
      },
      setOp: function(nt, Hn, bn) {
        (se !== nt || Le !== Hn || We !== bn) && (n.stencilOp(nt, Hn, bn), se = nt, Le = Hn, We = bn);
      },
      setLocked: function(nt) {
        F = nt;
      },
      setClear: function(nt) {
        ht !== nt && (n.clearStencil(nt), ht = nt);
      },
      reset: function() {
        F = !1, he = null, de = null, Re = null, ce = null, se = null, Le = null, We = null, ht = null;
      }
    };
  }
  const r = new t(), o = new i(), a = new s(), l = /* @__PURE__ */ new WeakMap(), c = /* @__PURE__ */ new WeakMap();
  let u = {}, h = {}, f = /* @__PURE__ */ new WeakMap(), p = [], v = null, x = !1, m = null, d = null, b = null, A = null, M = null, C = null, w = null, P = new Xe(0, 0, 0), U = 0, S = !1, y = null, D = null, L = null, V = null, Z = null;
  const ne = n.getParameter(n.MAX_COMBINED_TEXTURE_IMAGE_UNITS);
  let J = !1, ie = 0;
  const H = n.getParameter(n.VERSION);
  H.indexOf("WebGL") !== -1 ? (ie = parseFloat(/^WebGL (\d)/.exec(H)[1]), J = ie >= 1) : H.indexOf("OpenGL ES") !== -1 && (ie = parseFloat(/^OpenGL ES (\d)/.exec(H)[1]), J = ie >= 2);
  let fe = null, ge = {};
  const ye = n.getParameter(n.SCISSOR_BOX), Fe = n.getParameter(n.VIEWPORT), Je = new lt().fromArray(ye), Ge = new lt().fromArray(Fe);
  function Ae(F, he, de, Re) {
    const ce = new Uint8Array(4), se = n.createTexture();
    n.bindTexture(F, se), n.texParameteri(F, n.TEXTURE_MIN_FILTER, n.NEAREST), n.texParameteri(F, n.TEXTURE_MAG_FILTER, n.NEAREST);
    for (let Le = 0; Le < de; Le++)
      F === n.TEXTURE_3D || F === n.TEXTURE_2D_ARRAY ? n.texImage3D(he, 0, n.RGBA, 1, 1, Re, 0, n.RGBA, n.UNSIGNED_BYTE, ce) : n.texImage2D(he + Le, 0, n.RGBA, 1, 1, 0, n.RGBA, n.UNSIGNED_BYTE, ce);
    return se;
  }
  const X = {};
  X[n.TEXTURE_2D] = Ae(n.TEXTURE_2D, n.TEXTURE_2D, 1), X[n.TEXTURE_CUBE_MAP] = Ae(n.TEXTURE_CUBE_MAP, n.TEXTURE_CUBE_MAP_POSITIVE_X, 6), X[n.TEXTURE_2D_ARRAY] = Ae(n.TEXTURE_2D_ARRAY, n.TEXTURE_2D_ARRAY, 1, 1), X[n.TEXTURE_3D] = Ae(n.TEXTURE_3D, n.TEXTURE_3D, 1, 1), r.setClear(0, 0, 0, 1), o.setClear(1), a.setClear(0), re(n.DEPTH_TEST), o.setFunc(Ns), Y(!1), z(Au), re(n.CULL_FACE), W(xi);
  function re(F) {
    u[F] !== !0 && (n.enable(F), u[F] = !0);
  }
  function be(F) {
    u[F] !== !1 && (n.disable(F), u[F] = !1);
  }
  function Be(F, he) {
    return h[F] !== he ? (n.bindFramebuffer(F, he), h[F] = he, F === n.DRAW_FRAMEBUFFER && (h[n.FRAMEBUFFER] = he), F === n.FRAMEBUFFER && (h[n.DRAW_FRAMEBUFFER] = he), !0) : !1;
  }
  function Pe(F, he) {
    let de = p, Re = !1;
    if (F) {
      de = f.get(he), de === void 0 && (de = [], f.set(he, de));
      const ce = F.textures;
      if (de.length !== ce.length || de[0] !== n.COLOR_ATTACHMENT0) {
        for (let se = 0, Le = ce.length; se < Le; se++)
          de[se] = n.COLOR_ATTACHMENT0 + se;
        de.length = ce.length, Re = !0;
      }
    } else
      de[0] !== n.BACK && (de[0] = n.BACK, Re = !0);
    Re && n.drawBuffers(de);
  }
  function Ze(F) {
    return v !== F ? (n.useProgram(F), v = F, !0) : !1;
  }
  const R = {
    [zi]: n.FUNC_ADD,
    [K_]: n.FUNC_SUBTRACT,
    [$_]: n.FUNC_REVERSE_SUBTRACT
  };
  R[Z_] = n.MIN, R[J_] = n.MAX;
  const g = {
    [Q_]: n.ZERO,
    [eg]: n.ONE,
    [tg]: n.SRC_COLOR,
    [fl]: n.SRC_ALPHA,
    [ag]: n.SRC_ALPHA_SATURATE,
    [rg]: n.DST_COLOR,
    [ig]: n.DST_ALPHA,
    [ng]: n.ONE_MINUS_SRC_COLOR,
    [dl]: n.ONE_MINUS_SRC_ALPHA,
    [og]: n.ONE_MINUS_DST_COLOR,
    [sg]: n.ONE_MINUS_DST_ALPHA,
    [lg]: n.CONSTANT_COLOR,
    [cg]: n.ONE_MINUS_CONSTANT_COLOR,
    [ug]: n.CONSTANT_ALPHA,
    [hg]: n.ONE_MINUS_CONSTANT_ALPHA
  };
  function W(F, he, de, Re, ce, se, Le, We, ht, nt) {
    if (F === xi) {
      x === !0 && (be(n.BLEND), x = !1);
      return;
    }
    if (x === !1 && (re(n.BLEND), x = !0), F !== j_) {
      if (F !== m || nt !== S) {
        if ((d !== zi || M !== zi) && (n.blendEquation(n.FUNC_ADD), d = zi, M = zi), nt)
          switch (F) {
            case Ls:
              n.blendFuncSeparate(n.ONE, n.ONE_MINUS_SRC_ALPHA, n.ONE, n.ONE_MINUS_SRC_ALPHA);
              break;
            case wu:
              n.blendFunc(n.ONE, n.ONE);
              break;
            case Ru:
              n.blendFuncSeparate(n.ZERO, n.ONE_MINUS_SRC_COLOR, n.ZERO, n.ONE);
              break;
            case Cu:
              n.blendFuncSeparate(n.DST_COLOR, n.ONE_MINUS_SRC_ALPHA, n.ZERO, n.ONE);
              break;
            default:
              console.error("THREE.WebGLState: Invalid blending: ", F);
              break;
          }
        else
          switch (F) {
            case Ls:
              n.blendFuncSeparate(n.SRC_ALPHA, n.ONE_MINUS_SRC_ALPHA, n.ONE, n.ONE_MINUS_SRC_ALPHA);
              break;
            case wu:
              n.blendFuncSeparate(n.SRC_ALPHA, n.ONE, n.ONE, n.ONE);
              break;
            case Ru:
              console.error("THREE.WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");
              break;
            case Cu:
              console.error("THREE.WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");
              break;
            default:
              console.error("THREE.WebGLState: Invalid blending: ", F);
              break;
          }
        b = null, A = null, C = null, w = null, P.set(0, 0, 0), U = 0, m = F, S = nt;
      }
      return;
    }
    ce = ce || he, se = se || de, Le = Le || Re, (he !== d || ce !== M) && (n.blendEquationSeparate(R[he], R[ce]), d = he, M = ce), (de !== b || Re !== A || se !== C || Le !== w) && (n.blendFuncSeparate(g[de], g[Re], g[se], g[Le]), b = de, A = Re, C = se, w = Le), (We.equals(P) === !1 || ht !== U) && (n.blendColor(We.r, We.g, We.b, ht), P.copy(We), U = ht), m = F, S = !1;
  }
  function K(F, he) {
    F.side === Qn ? be(n.CULL_FACE) : re(n.CULL_FACE);
    let de = F.side === Wt;
    he && (de = !de), Y(de), F.blending === Ls && F.transparent === !1 ? W(xi) : W(F.blending, F.blendEquation, F.blendSrc, F.blendDst, F.blendEquationAlpha, F.blendSrcAlpha, F.blendDstAlpha, F.blendColor, F.blendAlpha, F.premultipliedAlpha), o.setFunc(F.depthFunc), o.setTest(F.depthTest), o.setMask(F.depthWrite), r.setMask(F.colorWrite);
    const Re = F.stencilWrite;
    a.setTest(Re), Re && (a.setMask(F.stencilWriteMask), a.setFunc(F.stencilFunc, F.stencilRef, F.stencilFuncMask), a.setOp(F.stencilFail, F.stencilZFail, F.stencilZPass)), j(F.polygonOffset, F.polygonOffsetFactor, F.polygonOffsetUnits), F.alphaToCoverage === !0 ? re(n.SAMPLE_ALPHA_TO_COVERAGE) : be(n.SAMPLE_ALPHA_TO_COVERAGE);
  }
  function Y(F) {
    y !== F && (F ? n.frontFace(n.CW) : n.frontFace(n.CCW), y = F);
  }
  function z(F) {
    F !== X_ ? (re(n.CULL_FACE), F !== D && (F === Au ? n.cullFace(n.BACK) : F === Y_ ? n.cullFace(n.FRONT) : n.cullFace(n.FRONT_AND_BACK))) : be(n.CULL_FACE), D = F;
  }
  function ae(F) {
    F !== L && (J && n.lineWidth(F), L = F);
  }
  function j(F, he, de) {
    F ? (re(n.POLYGON_OFFSET_FILL), (V !== he || Z !== de) && (n.polygonOffset(he, de), V = he, Z = de)) : be(n.POLYGON_OFFSET_FILL);
  }
  function ee(F) {
    F ? re(n.SCISSOR_TEST) : be(n.SCISSOR_TEST);
  }
  function te(F) {
    F === void 0 && (F = n.TEXTURE0 + ne - 1), fe !== F && (n.activeTexture(F), fe = F);
  }
  function xe(F, he, de) {
    de === void 0 && (fe === null ? de = n.TEXTURE0 + ne - 1 : de = fe);
    let Re = ge[de];
    Re === void 0 && (Re = { type: void 0, texture: void 0 }, ge[de] = Re), (Re.type !== F || Re.texture !== he) && (fe !== de && (n.activeTexture(de), fe = de), n.bindTexture(F, he || X[F]), Re.type = F, Re.texture = he);
  }
  function E() {
    const F = ge[fe];
    F !== void 0 && F.type !== void 0 && (n.bindTexture(F.type, null), F.type = void 0, F.texture = void 0);
  }
  function _() {
    try {
      n.compressedTexImage2D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function I() {
    try {
      n.compressedTexImage3D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function k() {
    try {
      n.texSubImage2D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function Q() {
    try {
      n.texSubImage3D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function G() {
    try {
      n.compressedTexSubImage2D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function pe() {
    try {
      n.compressedTexSubImage3D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function oe() {
    try {
      n.texStorage2D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function Se() {
    try {
      n.texStorage3D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function Ee() {
    try {
      n.texImage2D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function le() {
    try {
      n.texImage3D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function ve(F) {
    Je.equals(F) === !1 && (n.scissor(F.x, F.y, F.z, F.w), Je.copy(F));
  }
  function Ce(F) {
    Ge.equals(F) === !1 && (n.viewport(F.x, F.y, F.z, F.w), Ge.copy(F));
  }
  function Te(F, he) {
    let de = c.get(he);
    de === void 0 && (de = /* @__PURE__ */ new WeakMap(), c.set(he, de));
    let Re = de.get(F);
    Re === void 0 && (Re = n.getUniformBlockIndex(he, F.name), de.set(F, Re));
  }
  function me(F, he) {
    const Re = c.get(he).get(F);
    l.get(he) !== Re && (n.uniformBlockBinding(he, Re, F.__bindingPointIndex), l.set(he, Re));
  }
  function ke() {
    n.disable(n.BLEND), n.disable(n.CULL_FACE), n.disable(n.DEPTH_TEST), n.disable(n.POLYGON_OFFSET_FILL), n.disable(n.SCISSOR_TEST), n.disable(n.STENCIL_TEST), n.disable(n.SAMPLE_ALPHA_TO_COVERAGE), n.blendEquation(n.FUNC_ADD), n.blendFunc(n.ONE, n.ZERO), n.blendFuncSeparate(n.ONE, n.ZERO, n.ONE, n.ZERO), n.blendColor(0, 0, 0, 0), n.colorMask(!0, !0, !0, !0), n.clearColor(0, 0, 0, 0), n.depthMask(!0), n.depthFunc(n.LESS), o.setReversed(!1), n.clearDepth(1), n.stencilMask(4294967295), n.stencilFunc(n.ALWAYS, 0, 4294967295), n.stencilOp(n.KEEP, n.KEEP, n.KEEP), n.clearStencil(0), n.cullFace(n.BACK), n.frontFace(n.CCW), n.polygonOffset(0, 0), n.activeTexture(n.TEXTURE0), n.bindFramebuffer(n.FRAMEBUFFER, null), n.bindFramebuffer(n.DRAW_FRAMEBUFFER, null), n.bindFramebuffer(n.READ_FRAMEBUFFER, null), n.useProgram(null), n.lineWidth(1), n.scissor(0, 0, n.canvas.width, n.canvas.height), n.viewport(0, 0, n.canvas.width, n.canvas.height), u = {}, fe = null, ge = {}, h = {}, f = /* @__PURE__ */ new WeakMap(), p = [], v = null, x = !1, m = null, d = null, b = null, A = null, M = null, C = null, w = null, P = new Xe(0, 0, 0), U = 0, S = !1, y = null, D = null, L = null, V = null, Z = null, Je.set(0, 0, n.canvas.width, n.canvas.height), Ge.set(0, 0, n.canvas.width, n.canvas.height), r.reset(), o.reset(), a.reset();
  }
  return {
    buffers: {
      color: r,
      depth: o,
      stencil: a
    },
    enable: re,
    disable: be,
    bindFramebuffer: Be,
    drawBuffers: Pe,
    useProgram: Ze,
    setBlending: W,
    setMaterial: K,
    setFlipSided: Y,
    setCullFace: z,
    setLineWidth: ae,
    setPolygonOffset: j,
    setScissorTest: ee,
    activeTexture: te,
    bindTexture: xe,
    unbindTexture: E,
    compressedTexImage2D: _,
    compressedTexImage3D: I,
    texImage2D: Ee,
    texImage3D: le,
    updateUBOMapping: Te,
    uniformBlockBinding: me,
    texStorage2D: oe,
    texStorage3D: Se,
    texSubImage2D: k,
    texSubImage3D: Q,
    compressedTexSubImage2D: G,
    compressedTexSubImage3D: pe,
    scissor: ve,
    viewport: Ce,
    reset: ke
  };
}
function KS(n, e, t, i, s, r, o) {
  const a = e.has("WEBGL_multisampled_render_to_texture") ? e.get("WEBGL_multisampled_render_to_texture") : null, l = typeof navigator > "u" ? !1 : /OculusBrowser/g.test(navigator.userAgent), c = new Ve(), u = /* @__PURE__ */ new WeakMap();
  let h;
  const f = /* @__PURE__ */ new WeakMap();
  let p = !1;
  try {
    p = typeof OffscreenCanvas < "u" && new OffscreenCanvas(1, 1).getContext("2d") !== null;
  } catch {
  }
  function v(E, _) {
    return p ? (
      // eslint-disable-next-line compat/compat
      new OffscreenCanvas(E, _)
    ) : Ho("canvas");
  }
  function x(E, _, I) {
    let k = 1;
    const Q = xe(E);
    if ((Q.width > I || Q.height > I) && (k = I / Math.max(Q.width, Q.height)), k < 1)
      if (typeof HTMLImageElement < "u" && E instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && E instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && E instanceof ImageBitmap || typeof VideoFrame < "u" && E instanceof VideoFrame) {
        const G = Math.floor(k * Q.width), pe = Math.floor(k * Q.height);
        h === void 0 && (h = v(G, pe));
        const oe = _ ? v(G, pe) : h;
        return oe.width = G, oe.height = pe, oe.getContext("2d").drawImage(E, 0, 0, G, pe), console.warn("THREE.WebGLRenderer: Texture has been resized from (" + Q.width + "x" + Q.height + ") to (" + G + "x" + pe + ")."), oe;
      } else
        return "data" in E && console.warn("THREE.WebGLRenderer: Image in DataTexture is too big (" + Q.width + "x" + Q.height + ")."), E;
    return E;
  }
  function m(E) {
    return E.generateMipmaps;
  }
  function d(E) {
    n.generateMipmap(E);
  }
  function b(E) {
    return E.isWebGLCubeRenderTarget ? n.TEXTURE_CUBE_MAP : E.isWebGL3DRenderTarget ? n.TEXTURE_3D : E.isWebGLArrayRenderTarget || E.isCompressedArrayTexture ? n.TEXTURE_2D_ARRAY : n.TEXTURE_2D;
  }
  function A(E, _, I, k, Q = !1) {
    if (E !== null) {
      if (n[E] !== void 0) return n[E];
      console.warn("THREE.WebGLRenderer: Attempt to use non-existing WebGL internal format '" + E + "'");
    }
    let G = _;
    if (_ === n.RED && (I === n.FLOAT && (G = n.R32F), I === n.HALF_FLOAT && (G = n.R16F), I === n.UNSIGNED_BYTE && (G = n.R8)), _ === n.RED_INTEGER && (I === n.UNSIGNED_BYTE && (G = n.R8UI), I === n.UNSIGNED_SHORT && (G = n.R16UI), I === n.UNSIGNED_INT && (G = n.R32UI), I === n.BYTE && (G = n.R8I), I === n.SHORT && (G = n.R16I), I === n.INT && (G = n.R32I)), _ === n.RG && (I === n.FLOAT && (G = n.RG32F), I === n.HALF_FLOAT && (G = n.RG16F), I === n.UNSIGNED_BYTE && (G = n.RG8)), _ === n.RG_INTEGER && (I === n.UNSIGNED_BYTE && (G = n.RG8UI), I === n.UNSIGNED_SHORT && (G = n.RG16UI), I === n.UNSIGNED_INT && (G = n.RG32UI), I === n.BYTE && (G = n.RG8I), I === n.SHORT && (G = n.RG16I), I === n.INT && (G = n.RG32I)), _ === n.RGB_INTEGER && (I === n.UNSIGNED_BYTE && (G = n.RGB8UI), I === n.UNSIGNED_SHORT && (G = n.RGB16UI), I === n.UNSIGNED_INT && (G = n.RGB32UI), I === n.BYTE && (G = n.RGB8I), I === n.SHORT && (G = n.RGB16I), I === n.INT && (G = n.RGB32I)), _ === n.RGBA_INTEGER && (I === n.UNSIGNED_BYTE && (G = n.RGBA8UI), I === n.UNSIGNED_SHORT && (G = n.RGBA16UI), I === n.UNSIGNED_INT && (G = n.RGBA32UI), I === n.BYTE && (G = n.RGBA8I), I === n.SHORT && (G = n.RGBA16I), I === n.INT && (G = n.RGBA32I)), _ === n.RGB && (I === n.UNSIGNED_INT_5_9_9_9_REV && (G = n.RGB9_E5), I === n.UNSIGNED_INT_10F_11F_11F_REV && (G = n.R11F_G11F_B10F)), _ === n.RGBA) {
      const pe = Q ? Bo : et.getTransfer(k);
      I === n.FLOAT && (G = n.RGBA32F), I === n.HALF_FLOAT && (G = n.RGBA16F), I === n.UNSIGNED_BYTE && (G = pe === ot ? n.SRGB8_ALPHA8 : n.RGBA8), I === n.UNSIGNED_SHORT_4_4_4_4 && (G = n.RGBA4), I === n.UNSIGNED_SHORT_5_5_5_1 && (G = n.RGB5_A1);
    }
    return (G === n.R16F || G === n.R32F || G === n.RG16F || G === n.RG32F || G === n.RGBA16F || G === n.RGBA32F) && e.get("EXT_color_buffer_float"), G;
  }
  function M(E, _) {
    let I;
    return E ? _ === null || _ === Yi || _ === Ar ? I = n.DEPTH24_STENCIL8 : _ === ei ? I = n.DEPTH32F_STENCIL8 : _ === br && (I = n.DEPTH24_STENCIL8, console.warn("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")) : _ === null || _ === Yi || _ === Ar ? I = n.DEPTH_COMPONENT24 : _ === ei ? I = n.DEPTH_COMPONENT32F : _ === br && (I = n.DEPTH_COMPONENT16), I;
  }
  function C(E, _) {
    return m(E) === !0 || E.isFramebufferTexture && E.minFilter !== yn && E.minFilter !== Un ? Math.log2(Math.max(_.width, _.height)) + 1 : E.mipmaps !== void 0 && E.mipmaps.length > 0 ? E.mipmaps.length : E.isCompressedTexture && Array.isArray(E.image) ? _.mipmaps.length : 1;
  }
  function w(E) {
    const _ = E.target;
    _.removeEventListener("dispose", w), U(_), _.isVideoTexture && u.delete(_);
  }
  function P(E) {
    const _ = E.target;
    _.removeEventListener("dispose", P), y(_);
  }
  function U(E) {
    const _ = i.get(E);
    if (_.__webglInit === void 0) return;
    const I = E.source, k = f.get(I);
    if (k) {
      const Q = k[_.__cacheKey];
      Q.usedTimes--, Q.usedTimes === 0 && S(E), Object.keys(k).length === 0 && f.delete(I);
    }
    i.remove(E);
  }
  function S(E) {
    const _ = i.get(E);
    n.deleteTexture(_.__webglTexture);
    const I = E.source, k = f.get(I);
    delete k[_.__cacheKey], o.memory.textures--;
  }
  function y(E) {
    const _ = i.get(E);
    if (E.depthTexture && (E.depthTexture.dispose(), i.remove(E.depthTexture)), E.isWebGLCubeRenderTarget)
      for (let k = 0; k < 6; k++) {
        if (Array.isArray(_.__webglFramebuffer[k]))
          for (let Q = 0; Q < _.__webglFramebuffer[k].length; Q++) n.deleteFramebuffer(_.__webglFramebuffer[k][Q]);
        else
          n.deleteFramebuffer(_.__webglFramebuffer[k]);
        _.__webglDepthbuffer && n.deleteRenderbuffer(_.__webglDepthbuffer[k]);
      }
    else {
      if (Array.isArray(_.__webglFramebuffer))
        for (let k = 0; k < _.__webglFramebuffer.length; k++) n.deleteFramebuffer(_.__webglFramebuffer[k]);
      else
        n.deleteFramebuffer(_.__webglFramebuffer);
      if (_.__webglDepthbuffer && n.deleteRenderbuffer(_.__webglDepthbuffer), _.__webglMultisampledFramebuffer && n.deleteFramebuffer(_.__webglMultisampledFramebuffer), _.__webglColorRenderbuffer)
        for (let k = 0; k < _.__webglColorRenderbuffer.length; k++)
          _.__webglColorRenderbuffer[k] && n.deleteRenderbuffer(_.__webglColorRenderbuffer[k]);
      _.__webglDepthRenderbuffer && n.deleteRenderbuffer(_.__webglDepthRenderbuffer);
    }
    const I = E.textures;
    for (let k = 0, Q = I.length; k < Q; k++) {
      const G = i.get(I[k]);
      G.__webglTexture && (n.deleteTexture(G.__webglTexture), o.memory.textures--), i.remove(I[k]);
    }
    i.remove(E);
  }
  let D = 0;
  function L() {
    D = 0;
  }
  function V() {
    const E = D;
    return E >= s.maxTextures && console.warn("THREE.WebGLTextures: Trying to use " + E + " texture units while this GPU supports only " + s.maxTextures), D += 1, E;
  }
  function Z(E) {
    const _ = [];
    return _.push(E.wrapS), _.push(E.wrapT), _.push(E.wrapR || 0), _.push(E.magFilter), _.push(E.minFilter), _.push(E.anisotropy), _.push(E.internalFormat), _.push(E.format), _.push(E.type), _.push(E.generateMipmaps), _.push(E.premultiplyAlpha), _.push(E.flipY), _.push(E.unpackAlignment), _.push(E.colorSpace), _.join();
  }
  function ne(E, _) {
    const I = i.get(E);
    if (E.isVideoTexture && ee(E), E.isRenderTargetTexture === !1 && E.isExternalTexture !== !0 && E.version > 0 && I.__version !== E.version) {
      const k = E.image;
      if (k === null)
        console.warn("THREE.WebGLRenderer: Texture marked for update but no image data found.");
      else if (k.complete === !1)
        console.warn("THREE.WebGLRenderer: Texture marked for update but image is incomplete");
      else {
        X(I, E, _);
        return;
      }
    } else E.isExternalTexture && (I.__webglTexture = E.sourceTexture ? E.sourceTexture : null);
    t.bindTexture(n.TEXTURE_2D, I.__webglTexture, n.TEXTURE0 + _);
  }
  function J(E, _) {
    const I = i.get(E);
    if (E.isRenderTargetTexture === !1 && E.version > 0 && I.__version !== E.version) {
      X(I, E, _);
      return;
    }
    t.bindTexture(n.TEXTURE_2D_ARRAY, I.__webglTexture, n.TEXTURE0 + _);
  }
  function ie(E, _) {
    const I = i.get(E);
    if (E.isRenderTargetTexture === !1 && E.version > 0 && I.__version !== E.version) {
      X(I, E, _);
      return;
    }
    t.bindTexture(n.TEXTURE_3D, I.__webglTexture, n.TEXTURE0 + _);
  }
  function H(E, _) {
    const I = i.get(E);
    if (E.version > 0 && I.__version !== E.version) {
      re(I, E, _);
      return;
    }
    t.bindTexture(n.TEXTURE_CUBE_MAP, I.__webglTexture, n.TEXTURE0 + _);
  }
  const fe = {
    [El]: n.REPEAT,
    [Vi]: n.CLAMP_TO_EDGE,
    [Tl]: n.MIRRORED_REPEAT
  }, ge = {
    [yn]: n.NEAREST,
    [Mg]: n.NEAREST_MIPMAP_NEAREST,
    [kr]: n.NEAREST_MIPMAP_LINEAR,
    [Un]: n.LINEAR,
    [xa]: n.LINEAR_MIPMAP_NEAREST,
    [ki]: n.LINEAR_MIPMAP_LINEAR
  }, ye = {
    [Tg]: n.NEVER,
    [Pg]: n.ALWAYS,
    [bg]: n.LESS,
    [dd]: n.LEQUAL,
    [Ag]: n.EQUAL,
    [Cg]: n.GEQUAL,
    [wg]: n.GREATER,
    [Rg]: n.NOTEQUAL
  };
  function Fe(E, _) {
    if (_.type === ei && e.has("OES_texture_float_linear") === !1 && (_.magFilter === Un || _.magFilter === xa || _.magFilter === kr || _.magFilter === ki || _.minFilter === Un || _.minFilter === xa || _.minFilter === kr || _.minFilter === ki) && console.warn("THREE.WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."), n.texParameteri(E, n.TEXTURE_WRAP_S, fe[_.wrapS]), n.texParameteri(E, n.TEXTURE_WRAP_T, fe[_.wrapT]), (E === n.TEXTURE_3D || E === n.TEXTURE_2D_ARRAY) && n.texParameteri(E, n.TEXTURE_WRAP_R, fe[_.wrapR]), n.texParameteri(E, n.TEXTURE_MAG_FILTER, ge[_.magFilter]), n.texParameteri(E, n.TEXTURE_MIN_FILTER, ge[_.minFilter]), _.compareFunction && (n.texParameteri(E, n.TEXTURE_COMPARE_MODE, n.COMPARE_REF_TO_TEXTURE), n.texParameteri(E, n.TEXTURE_COMPARE_FUNC, ye[_.compareFunction])), e.has("EXT_texture_filter_anisotropic") === !0) {
      if (_.magFilter === yn || _.minFilter !== kr && _.minFilter !== ki || _.type === ei && e.has("OES_texture_float_linear") === !1) return;
      if (_.anisotropy > 1 || i.get(_).__currentAnisotropy) {
        const I = e.get("EXT_texture_filter_anisotropic");
        n.texParameterf(E, I.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(_.anisotropy, s.getMaxAnisotropy())), i.get(_).__currentAnisotropy = _.anisotropy;
      }
    }
  }
  function Je(E, _) {
    let I = !1;
    E.__webglInit === void 0 && (E.__webglInit = !0, _.addEventListener("dispose", w));
    const k = _.source;
    let Q = f.get(k);
    Q === void 0 && (Q = {}, f.set(k, Q));
    const G = Z(_);
    if (G !== E.__cacheKey) {
      Q[G] === void 0 && (Q[G] = {
        texture: n.createTexture(),
        usedTimes: 0
      }, o.memory.textures++, I = !0), Q[G].usedTimes++;
      const pe = Q[E.__cacheKey];
      pe !== void 0 && (Q[E.__cacheKey].usedTimes--, pe.usedTimes === 0 && S(_)), E.__cacheKey = G, E.__webglTexture = Q[G].texture;
    }
    return I;
  }
  function Ge(E, _, I) {
    return Math.floor(Math.floor(E / I) / _);
  }
  function Ae(E, _, I, k) {
    const G = E.updateRanges;
    if (G.length === 0)
      t.texSubImage2D(n.TEXTURE_2D, 0, 0, 0, _.width, _.height, I, k, _.data);
    else {
      G.sort((le, ve) => le.start - ve.start);
      let pe = 0;
      for (let le = 1; le < G.length; le++) {
        const ve = G[pe], Ce = G[le], Te = ve.start + ve.count, me = Ge(Ce.start, _.width, 4), ke = Ge(ve.start, _.width, 4);
        Ce.start <= Te + 1 && me === ke && Ge(Ce.start + Ce.count - 1, _.width, 4) === me ? ve.count = Math.max(
          ve.count,
          Ce.start + Ce.count - ve.start
        ) : (++pe, G[pe] = Ce);
      }
      G.length = pe + 1;
      const oe = n.getParameter(n.UNPACK_ROW_LENGTH), Se = n.getParameter(n.UNPACK_SKIP_PIXELS), Ee = n.getParameter(n.UNPACK_SKIP_ROWS);
      n.pixelStorei(n.UNPACK_ROW_LENGTH, _.width);
      for (let le = 0, ve = G.length; le < ve; le++) {
        const Ce = G[le], Te = Math.floor(Ce.start / 4), me = Math.ceil(Ce.count / 4), ke = Te % _.width, F = Math.floor(Te / _.width), he = me, de = 1;
        n.pixelStorei(n.UNPACK_SKIP_PIXELS, ke), n.pixelStorei(n.UNPACK_SKIP_ROWS, F), t.texSubImage2D(n.TEXTURE_2D, 0, ke, F, he, de, I, k, _.data);
      }
      E.clearUpdateRanges(), n.pixelStorei(n.UNPACK_ROW_LENGTH, oe), n.pixelStorei(n.UNPACK_SKIP_PIXELS, Se), n.pixelStorei(n.UNPACK_SKIP_ROWS, Ee);
    }
  }
  function X(E, _, I) {
    let k = n.TEXTURE_2D;
    (_.isDataArrayTexture || _.isCompressedArrayTexture) && (k = n.TEXTURE_2D_ARRAY), _.isData3DTexture && (k = n.TEXTURE_3D);
    const Q = Je(E, _), G = _.source;
    t.bindTexture(k, E.__webglTexture, n.TEXTURE0 + I);
    const pe = i.get(G);
    if (G.version !== pe.__version || Q === !0) {
      t.activeTexture(n.TEXTURE0 + I);
      const oe = et.getPrimaries(et.workingColorSpace), Se = _.colorSpace === vi ? null : et.getPrimaries(_.colorSpace), Ee = _.colorSpace === vi || oe === Se ? n.NONE : n.BROWSER_DEFAULT_WEBGL;
      n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL, _.flipY), n.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL, _.premultiplyAlpha), n.pixelStorei(n.UNPACK_ALIGNMENT, _.unpackAlignment), n.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL, Ee);
      let le = x(_.image, !1, s.maxTextureSize);
      le = te(_, le);
      const ve = r.convert(_.format, _.colorSpace), Ce = r.convert(_.type);
      let Te = A(_.internalFormat, ve, Ce, _.colorSpace, _.isVideoTexture);
      Fe(k, _);
      let me;
      const ke = _.mipmaps, F = _.isVideoTexture !== !0, he = pe.__version === void 0 || Q === !0, de = G.dataReady, Re = C(_, le);
      if (_.isDepthTexture)
        Te = M(_.format === Rr, _.type), he && (F ? t.texStorage2D(n.TEXTURE_2D, 1, Te, le.width, le.height) : t.texImage2D(n.TEXTURE_2D, 0, Te, le.width, le.height, 0, ve, Ce, null));
      else if (_.isDataTexture)
        if (ke.length > 0) {
          F && he && t.texStorage2D(n.TEXTURE_2D, Re, Te, ke[0].width, ke[0].height);
          for (let ce = 0, se = ke.length; ce < se; ce++)
            me = ke[ce], F ? de && t.texSubImage2D(n.TEXTURE_2D, ce, 0, 0, me.width, me.height, ve, Ce, me.data) : t.texImage2D(n.TEXTURE_2D, ce, Te, me.width, me.height, 0, ve, Ce, me.data);
          _.generateMipmaps = !1;
        } else
          F ? (he && t.texStorage2D(n.TEXTURE_2D, Re, Te, le.width, le.height), de && Ae(_, le, ve, Ce)) : t.texImage2D(n.TEXTURE_2D, 0, Te, le.width, le.height, 0, ve, Ce, le.data);
      else if (_.isCompressedTexture)
        if (_.isCompressedArrayTexture) {
          F && he && t.texStorage3D(n.TEXTURE_2D_ARRAY, Re, Te, ke[0].width, ke[0].height, le.depth);
          for (let ce = 0, se = ke.length; ce < se; ce++)
            if (me = ke[ce], _.format !== xn)
              if (ve !== null)
                if (F) {
                  if (de)
                    if (_.layerUpdates.size > 0) {
                      const Le = lh(me.width, me.height, _.format, _.type);
                      for (const We of _.layerUpdates) {
                        const ht = me.data.subarray(
                          We * Le / me.data.BYTES_PER_ELEMENT,
                          (We + 1) * Le / me.data.BYTES_PER_ELEMENT
                        );
                        t.compressedTexSubImage3D(n.TEXTURE_2D_ARRAY, ce, 0, 0, We, me.width, me.height, 1, ve, ht);
                      }
                      _.clearLayerUpdates();
                    } else
                      t.compressedTexSubImage3D(n.TEXTURE_2D_ARRAY, ce, 0, 0, 0, me.width, me.height, le.depth, ve, me.data);
                } else
                  t.compressedTexImage3D(n.TEXTURE_2D_ARRAY, ce, Te, me.width, me.height, le.depth, 0, me.data, 0, 0);
              else
                console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");
            else
              F ? de && t.texSubImage3D(n.TEXTURE_2D_ARRAY, ce, 0, 0, 0, me.width, me.height, le.depth, ve, Ce, me.data) : t.texImage3D(n.TEXTURE_2D_ARRAY, ce, Te, me.width, me.height, le.depth, 0, ve, Ce, me.data);
        } else {
          F && he && t.texStorage2D(n.TEXTURE_2D, Re, Te, ke[0].width, ke[0].height);
          for (let ce = 0, se = ke.length; ce < se; ce++)
            me = ke[ce], _.format !== xn ? ve !== null ? F ? de && t.compressedTexSubImage2D(n.TEXTURE_2D, ce, 0, 0, me.width, me.height, ve, me.data) : t.compressedTexImage2D(n.TEXTURE_2D, ce, Te, me.width, me.height, 0, me.data) : console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()") : F ? de && t.texSubImage2D(n.TEXTURE_2D, ce, 0, 0, me.width, me.height, ve, Ce, me.data) : t.texImage2D(n.TEXTURE_2D, ce, Te, me.width, me.height, 0, ve, Ce, me.data);
        }
      else if (_.isDataArrayTexture)
        if (F) {
          if (he && t.texStorage3D(n.TEXTURE_2D_ARRAY, Re, Te, le.width, le.height, le.depth), de)
            if (_.layerUpdates.size > 0) {
              const ce = lh(le.width, le.height, _.format, _.type);
              for (const se of _.layerUpdates) {
                const Le = le.data.subarray(
                  se * ce / le.data.BYTES_PER_ELEMENT,
                  (se + 1) * ce / le.data.BYTES_PER_ELEMENT
                );
                t.texSubImage3D(n.TEXTURE_2D_ARRAY, 0, 0, 0, se, le.width, le.height, 1, ve, Ce, Le);
              }
              _.clearLayerUpdates();
            } else
              t.texSubImage3D(n.TEXTURE_2D_ARRAY, 0, 0, 0, 0, le.width, le.height, le.depth, ve, Ce, le.data);
        } else
          t.texImage3D(n.TEXTURE_2D_ARRAY, 0, Te, le.width, le.height, le.depth, 0, ve, Ce, le.data);
      else if (_.isData3DTexture)
        F ? (he && t.texStorage3D(n.TEXTURE_3D, Re, Te, le.width, le.height, le.depth), de && t.texSubImage3D(n.TEXTURE_3D, 0, 0, 0, 0, le.width, le.height, le.depth, ve, Ce, le.data)) : t.texImage3D(n.TEXTURE_3D, 0, Te, le.width, le.height, le.depth, 0, ve, Ce, le.data);
      else if (_.isFramebufferTexture) {
        if (he)
          if (F)
            t.texStorage2D(n.TEXTURE_2D, Re, Te, le.width, le.height);
          else {
            let ce = le.width, se = le.height;
            for (let Le = 0; Le < Re; Le++)
              t.texImage2D(n.TEXTURE_2D, Le, Te, ce, se, 0, ve, Ce, null), ce >>= 1, se >>= 1;
          }
      } else if (ke.length > 0) {
        if (F && he) {
          const ce = xe(ke[0]);
          t.texStorage2D(n.TEXTURE_2D, Re, Te, ce.width, ce.height);
        }
        for (let ce = 0, se = ke.length; ce < se; ce++)
          me = ke[ce], F ? de && t.texSubImage2D(n.TEXTURE_2D, ce, 0, 0, ve, Ce, me) : t.texImage2D(n.TEXTURE_2D, ce, Te, ve, Ce, me);
        _.generateMipmaps = !1;
      } else if (F) {
        if (he) {
          const ce = xe(le);
          t.texStorage2D(n.TEXTURE_2D, Re, Te, ce.width, ce.height);
        }
        de && t.texSubImage2D(n.TEXTURE_2D, 0, 0, 0, ve, Ce, le);
      } else
        t.texImage2D(n.TEXTURE_2D, 0, Te, ve, Ce, le);
      m(_) && d(k), pe.__version = G.version, _.onUpdate && _.onUpdate(_);
    }
    E.__version = _.version;
  }
  function re(E, _, I) {
    if (_.image.length !== 6) return;
    const k = Je(E, _), Q = _.source;
    t.bindTexture(n.TEXTURE_CUBE_MAP, E.__webglTexture, n.TEXTURE0 + I);
    const G = i.get(Q);
    if (Q.version !== G.__version || k === !0) {
      t.activeTexture(n.TEXTURE0 + I);
      const pe = et.getPrimaries(et.workingColorSpace), oe = _.colorSpace === vi ? null : et.getPrimaries(_.colorSpace), Se = _.colorSpace === vi || pe === oe ? n.NONE : n.BROWSER_DEFAULT_WEBGL;
      n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL, _.flipY), n.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL, _.premultiplyAlpha), n.pixelStorei(n.UNPACK_ALIGNMENT, _.unpackAlignment), n.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL, Se);
      const Ee = _.isCompressedTexture || _.image[0].isCompressedTexture, le = _.image[0] && _.image[0].isDataTexture, ve = [];
      for (let se = 0; se < 6; se++)
        !Ee && !le ? ve[se] = x(_.image[se], !0, s.maxCubemapSize) : ve[se] = le ? _.image[se].image : _.image[se], ve[se] = te(_, ve[se]);
      const Ce = ve[0], Te = r.convert(_.format, _.colorSpace), me = r.convert(_.type), ke = A(_.internalFormat, Te, me, _.colorSpace), F = _.isVideoTexture !== !0, he = G.__version === void 0 || k === !0, de = Q.dataReady;
      let Re = C(_, Ce);
      Fe(n.TEXTURE_CUBE_MAP, _);
      let ce;
      if (Ee) {
        F && he && t.texStorage2D(n.TEXTURE_CUBE_MAP, Re, ke, Ce.width, Ce.height);
        for (let se = 0; se < 6; se++) {
          ce = ve[se].mipmaps;
          for (let Le = 0; Le < ce.length; Le++) {
            const We = ce[Le];
            _.format !== xn ? Te !== null ? F ? de && t.compressedTexSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Le, 0, 0, We.width, We.height, Te, We.data) : t.compressedTexImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Le, ke, We.width, We.height, 0, We.data) : console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()") : F ? de && t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Le, 0, 0, We.width, We.height, Te, me, We.data) : t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Le, ke, We.width, We.height, 0, Te, me, We.data);
          }
        }
      } else {
        if (ce = _.mipmaps, F && he) {
          ce.length > 0 && Re++;
          const se = xe(ve[0]);
          t.texStorage2D(n.TEXTURE_CUBE_MAP, Re, ke, se.width, se.height);
        }
        for (let se = 0; se < 6; se++)
          if (le) {
            F ? de && t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, 0, 0, 0, ve[se].width, ve[se].height, Te, me, ve[se].data) : t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, 0, ke, ve[se].width, ve[se].height, 0, Te, me, ve[se].data);
            for (let Le = 0; Le < ce.length; Le++) {
              const ht = ce[Le].image[se].image;
              F ? de && t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Le + 1, 0, 0, ht.width, ht.height, Te, me, ht.data) : t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Le + 1, ke, ht.width, ht.height, 0, Te, me, ht.data);
            }
          } else {
            F ? de && t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, 0, 0, 0, Te, me, ve[se]) : t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, 0, ke, Te, me, ve[se]);
            for (let Le = 0; Le < ce.length; Le++) {
              const We = ce[Le];
              F ? de && t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Le + 1, 0, 0, Te, me, We.image[se]) : t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Le + 1, ke, Te, me, We.image[se]);
            }
          }
      }
      m(_) && d(n.TEXTURE_CUBE_MAP), G.__version = Q.version, _.onUpdate && _.onUpdate(_);
    }
    E.__version = _.version;
  }
  function be(E, _, I, k, Q, G) {
    const pe = r.convert(I.format, I.colorSpace), oe = r.convert(I.type), Se = A(I.internalFormat, pe, oe, I.colorSpace), Ee = i.get(_), le = i.get(I);
    if (le.__renderTarget = _, !Ee.__hasExternalTextures) {
      const ve = Math.max(1, _.width >> G), Ce = Math.max(1, _.height >> G);
      Q === n.TEXTURE_3D || Q === n.TEXTURE_2D_ARRAY ? t.texImage3D(Q, G, Se, ve, Ce, _.depth, 0, pe, oe, null) : t.texImage2D(Q, G, Se, ve, Ce, 0, pe, oe, null);
    }
    t.bindFramebuffer(n.FRAMEBUFFER, E), j(_) ? a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER, k, Q, le.__webglTexture, 0, ae(_)) : (Q === n.TEXTURE_2D || Q >= n.TEXTURE_CUBE_MAP_POSITIVE_X && Q <= n.TEXTURE_CUBE_MAP_NEGATIVE_Z) && n.framebufferTexture2D(n.FRAMEBUFFER, k, Q, le.__webglTexture, G), t.bindFramebuffer(n.FRAMEBUFFER, null);
  }
  function Be(E, _, I) {
    if (n.bindRenderbuffer(n.RENDERBUFFER, E), _.depthBuffer) {
      const k = _.depthTexture, Q = k && k.isDepthTexture ? k.type : null, G = M(_.stencilBuffer, Q), pe = _.stencilBuffer ? n.DEPTH_STENCIL_ATTACHMENT : n.DEPTH_ATTACHMENT, oe = ae(_);
      j(_) ? a.renderbufferStorageMultisampleEXT(n.RENDERBUFFER, oe, G, _.width, _.height) : I ? n.renderbufferStorageMultisample(n.RENDERBUFFER, oe, G, _.width, _.height) : n.renderbufferStorage(n.RENDERBUFFER, G, _.width, _.height), n.framebufferRenderbuffer(n.FRAMEBUFFER, pe, n.RENDERBUFFER, E);
    } else {
      const k = _.textures;
      for (let Q = 0; Q < k.length; Q++) {
        const G = k[Q], pe = r.convert(G.format, G.colorSpace), oe = r.convert(G.type), Se = A(G.internalFormat, pe, oe, G.colorSpace), Ee = ae(_);
        I && j(_) === !1 ? n.renderbufferStorageMultisample(n.RENDERBUFFER, Ee, Se, _.width, _.height) : j(_) ? a.renderbufferStorageMultisampleEXT(n.RENDERBUFFER, Ee, Se, _.width, _.height) : n.renderbufferStorage(n.RENDERBUFFER, Se, _.width, _.height);
      }
    }
    n.bindRenderbuffer(n.RENDERBUFFER, null);
  }
  function Pe(E, _) {
    if (_ && _.isWebGLCubeRenderTarget) throw new Error("Depth Texture with cube render targets is not supported");
    if (t.bindFramebuffer(n.FRAMEBUFFER, E), !(_.depthTexture && _.depthTexture.isDepthTexture))
      throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");
    const k = i.get(_.depthTexture);
    k.__renderTarget = _, (!k.__webglTexture || _.depthTexture.image.width !== _.width || _.depthTexture.image.height !== _.height) && (_.depthTexture.image.width = _.width, _.depthTexture.image.height = _.height, _.depthTexture.needsUpdate = !0), ne(_.depthTexture, 0);
    const Q = k.__webglTexture, G = ae(_);
    if (_.depthTexture.format === wr)
      j(_) ? a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER, n.DEPTH_ATTACHMENT, n.TEXTURE_2D, Q, 0, G) : n.framebufferTexture2D(n.FRAMEBUFFER, n.DEPTH_ATTACHMENT, n.TEXTURE_2D, Q, 0);
    else if (_.depthTexture.format === Rr)
      j(_) ? a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER, n.DEPTH_STENCIL_ATTACHMENT, n.TEXTURE_2D, Q, 0, G) : n.framebufferTexture2D(n.FRAMEBUFFER, n.DEPTH_STENCIL_ATTACHMENT, n.TEXTURE_2D, Q, 0);
    else
      throw new Error("Unknown depthTexture format");
  }
  function Ze(E) {
    const _ = i.get(E), I = E.isWebGLCubeRenderTarget === !0;
    if (_.__boundDepthTexture !== E.depthTexture) {
      const k = E.depthTexture;
      if (_.__depthDisposeCallback && _.__depthDisposeCallback(), k) {
        const Q = () => {
          delete _.__boundDepthTexture, delete _.__depthDisposeCallback, k.removeEventListener("dispose", Q);
        };
        k.addEventListener("dispose", Q), _.__depthDisposeCallback = Q;
      }
      _.__boundDepthTexture = k;
    }
    if (E.depthTexture && !_.__autoAllocateDepthBuffer) {
      if (I) throw new Error("target.depthTexture not supported in Cube render targets");
      const k = E.texture.mipmaps;
      k && k.length > 0 ? Pe(_.__webglFramebuffer[0], E) : Pe(_.__webglFramebuffer, E);
    } else if (I) {
      _.__webglDepthbuffer = [];
      for (let k = 0; k < 6; k++)
        if (t.bindFramebuffer(n.FRAMEBUFFER, _.__webglFramebuffer[k]), _.__webglDepthbuffer[k] === void 0)
          _.__webglDepthbuffer[k] = n.createRenderbuffer(), Be(_.__webglDepthbuffer[k], E, !1);
        else {
          const Q = E.stencilBuffer ? n.DEPTH_STENCIL_ATTACHMENT : n.DEPTH_ATTACHMENT, G = _.__webglDepthbuffer[k];
          n.bindRenderbuffer(n.RENDERBUFFER, G), n.framebufferRenderbuffer(n.FRAMEBUFFER, Q, n.RENDERBUFFER, G);
        }
    } else {
      const k = E.texture.mipmaps;
      if (k && k.length > 0 ? t.bindFramebuffer(n.FRAMEBUFFER, _.__webglFramebuffer[0]) : t.bindFramebuffer(n.FRAMEBUFFER, _.__webglFramebuffer), _.__webglDepthbuffer === void 0)
        _.__webglDepthbuffer = n.createRenderbuffer(), Be(_.__webglDepthbuffer, E, !1);
      else {
        const Q = E.stencilBuffer ? n.DEPTH_STENCIL_ATTACHMENT : n.DEPTH_ATTACHMENT, G = _.__webglDepthbuffer;
        n.bindRenderbuffer(n.RENDERBUFFER, G), n.framebufferRenderbuffer(n.FRAMEBUFFER, Q, n.RENDERBUFFER, G);
      }
    }
    t.bindFramebuffer(n.FRAMEBUFFER, null);
  }
  function R(E, _, I) {
    const k = i.get(E);
    _ !== void 0 && be(k.__webglFramebuffer, E, E.texture, n.COLOR_ATTACHMENT0, n.TEXTURE_2D, 0), I !== void 0 && Ze(E);
  }
  function g(E) {
    const _ = E.texture, I = i.get(E), k = i.get(_);
    E.addEventListener("dispose", P);
    const Q = E.textures, G = E.isWebGLCubeRenderTarget === !0, pe = Q.length > 1;
    if (pe || (k.__webglTexture === void 0 && (k.__webglTexture = n.createTexture()), k.__version = _.version, o.memory.textures++), G) {
      I.__webglFramebuffer = [];
      for (let oe = 0; oe < 6; oe++)
        if (_.mipmaps && _.mipmaps.length > 0) {
          I.__webglFramebuffer[oe] = [];
          for (let Se = 0; Se < _.mipmaps.length; Se++)
            I.__webglFramebuffer[oe][Se] = n.createFramebuffer();
        } else
          I.__webglFramebuffer[oe] = n.createFramebuffer();
    } else {
      if (_.mipmaps && _.mipmaps.length > 0) {
        I.__webglFramebuffer = [];
        for (let oe = 0; oe < _.mipmaps.length; oe++)
          I.__webglFramebuffer[oe] = n.createFramebuffer();
      } else
        I.__webglFramebuffer = n.createFramebuffer();
      if (pe)
        for (let oe = 0, Se = Q.length; oe < Se; oe++) {
          const Ee = i.get(Q[oe]);
          Ee.__webglTexture === void 0 && (Ee.__webglTexture = n.createTexture(), o.memory.textures++);
        }
      if (E.samples > 0 && j(E) === !1) {
        I.__webglMultisampledFramebuffer = n.createFramebuffer(), I.__webglColorRenderbuffer = [], t.bindFramebuffer(n.FRAMEBUFFER, I.__webglMultisampledFramebuffer);
        for (let oe = 0; oe < Q.length; oe++) {
          const Se = Q[oe];
          I.__webglColorRenderbuffer[oe] = n.createRenderbuffer(), n.bindRenderbuffer(n.RENDERBUFFER, I.__webglColorRenderbuffer[oe]);
          const Ee = r.convert(Se.format, Se.colorSpace), le = r.convert(Se.type), ve = A(Se.internalFormat, Ee, le, Se.colorSpace, E.isXRRenderTarget === !0), Ce = ae(E);
          n.renderbufferStorageMultisample(n.RENDERBUFFER, Ce, ve, E.width, E.height), n.framebufferRenderbuffer(n.FRAMEBUFFER, n.COLOR_ATTACHMENT0 + oe, n.RENDERBUFFER, I.__webglColorRenderbuffer[oe]);
        }
        n.bindRenderbuffer(n.RENDERBUFFER, null), E.depthBuffer && (I.__webglDepthRenderbuffer = n.createRenderbuffer(), Be(I.__webglDepthRenderbuffer, E, !0)), t.bindFramebuffer(n.FRAMEBUFFER, null);
      }
    }
    if (G) {
      t.bindTexture(n.TEXTURE_CUBE_MAP, k.__webglTexture), Fe(n.TEXTURE_CUBE_MAP, _);
      for (let oe = 0; oe < 6; oe++)
        if (_.mipmaps && _.mipmaps.length > 0)
          for (let Se = 0; Se < _.mipmaps.length; Se++)
            be(I.__webglFramebuffer[oe][Se], E, _, n.COLOR_ATTACHMENT0, n.TEXTURE_CUBE_MAP_POSITIVE_X + oe, Se);
        else
          be(I.__webglFramebuffer[oe], E, _, n.COLOR_ATTACHMENT0, n.TEXTURE_CUBE_MAP_POSITIVE_X + oe, 0);
      m(_) && d(n.TEXTURE_CUBE_MAP), t.unbindTexture();
    } else if (pe) {
      for (let oe = 0, Se = Q.length; oe < Se; oe++) {
        const Ee = Q[oe], le = i.get(Ee);
        let ve = n.TEXTURE_2D;
        (E.isWebGL3DRenderTarget || E.isWebGLArrayRenderTarget) && (ve = E.isWebGL3DRenderTarget ? n.TEXTURE_3D : n.TEXTURE_2D_ARRAY), t.bindTexture(ve, le.__webglTexture), Fe(ve, Ee), be(I.__webglFramebuffer, E, Ee, n.COLOR_ATTACHMENT0 + oe, ve, 0), m(Ee) && d(ve);
      }
      t.unbindTexture();
    } else {
      let oe = n.TEXTURE_2D;
      if ((E.isWebGL3DRenderTarget || E.isWebGLArrayRenderTarget) && (oe = E.isWebGL3DRenderTarget ? n.TEXTURE_3D : n.TEXTURE_2D_ARRAY), t.bindTexture(oe, k.__webglTexture), Fe(oe, _), _.mipmaps && _.mipmaps.length > 0)
        for (let Se = 0; Se < _.mipmaps.length; Se++)
          be(I.__webglFramebuffer[Se], E, _, n.COLOR_ATTACHMENT0, oe, Se);
      else
        be(I.__webglFramebuffer, E, _, n.COLOR_ATTACHMENT0, oe, 0);
      m(_) && d(oe), t.unbindTexture();
    }
    E.depthBuffer && Ze(E);
  }
  function W(E) {
    const _ = E.textures;
    for (let I = 0, k = _.length; I < k; I++) {
      const Q = _[I];
      if (m(Q)) {
        const G = b(E), pe = i.get(Q).__webglTexture;
        t.bindTexture(G, pe), d(G), t.unbindTexture();
      }
    }
  }
  const K = [], Y = [];
  function z(E) {
    if (E.samples > 0) {
      if (j(E) === !1) {
        const _ = E.textures, I = E.width, k = E.height;
        let Q = n.COLOR_BUFFER_BIT;
        const G = E.stencilBuffer ? n.DEPTH_STENCIL_ATTACHMENT : n.DEPTH_ATTACHMENT, pe = i.get(E), oe = _.length > 1;
        if (oe)
          for (let Ee = 0; Ee < _.length; Ee++)
            t.bindFramebuffer(n.FRAMEBUFFER, pe.__webglMultisampledFramebuffer), n.framebufferRenderbuffer(n.FRAMEBUFFER, n.COLOR_ATTACHMENT0 + Ee, n.RENDERBUFFER, null), t.bindFramebuffer(n.FRAMEBUFFER, pe.__webglFramebuffer), n.framebufferTexture2D(n.DRAW_FRAMEBUFFER, n.COLOR_ATTACHMENT0 + Ee, n.TEXTURE_2D, null, 0);
        t.bindFramebuffer(n.READ_FRAMEBUFFER, pe.__webglMultisampledFramebuffer);
        const Se = E.texture.mipmaps;
        Se && Se.length > 0 ? t.bindFramebuffer(n.DRAW_FRAMEBUFFER, pe.__webglFramebuffer[0]) : t.bindFramebuffer(n.DRAW_FRAMEBUFFER, pe.__webglFramebuffer);
        for (let Ee = 0; Ee < _.length; Ee++) {
          if (E.resolveDepthBuffer && (E.depthBuffer && (Q |= n.DEPTH_BUFFER_BIT), E.stencilBuffer && E.resolveStencilBuffer && (Q |= n.STENCIL_BUFFER_BIT)), oe) {
            n.framebufferRenderbuffer(n.READ_FRAMEBUFFER, n.COLOR_ATTACHMENT0, n.RENDERBUFFER, pe.__webglColorRenderbuffer[Ee]);
            const le = i.get(_[Ee]).__webglTexture;
            n.framebufferTexture2D(n.DRAW_FRAMEBUFFER, n.COLOR_ATTACHMENT0, n.TEXTURE_2D, le, 0);
          }
          n.blitFramebuffer(0, 0, I, k, 0, 0, I, k, Q, n.NEAREST), l === !0 && (K.length = 0, Y.length = 0, K.push(n.COLOR_ATTACHMENT0 + Ee), E.depthBuffer && E.resolveDepthBuffer === !1 && (K.push(G), Y.push(G), n.invalidateFramebuffer(n.DRAW_FRAMEBUFFER, Y)), n.invalidateFramebuffer(n.READ_FRAMEBUFFER, K));
        }
        if (t.bindFramebuffer(n.READ_FRAMEBUFFER, null), t.bindFramebuffer(n.DRAW_FRAMEBUFFER, null), oe)
          for (let Ee = 0; Ee < _.length; Ee++) {
            t.bindFramebuffer(n.FRAMEBUFFER, pe.__webglMultisampledFramebuffer), n.framebufferRenderbuffer(n.FRAMEBUFFER, n.COLOR_ATTACHMENT0 + Ee, n.RENDERBUFFER, pe.__webglColorRenderbuffer[Ee]);
            const le = i.get(_[Ee]).__webglTexture;
            t.bindFramebuffer(n.FRAMEBUFFER, pe.__webglFramebuffer), n.framebufferTexture2D(n.DRAW_FRAMEBUFFER, n.COLOR_ATTACHMENT0 + Ee, n.TEXTURE_2D, le, 0);
          }
        t.bindFramebuffer(n.DRAW_FRAMEBUFFER, pe.__webglMultisampledFramebuffer);
      } else if (E.depthBuffer && E.resolveDepthBuffer === !1 && l) {
        const _ = E.stencilBuffer ? n.DEPTH_STENCIL_ATTACHMENT : n.DEPTH_ATTACHMENT;
        n.invalidateFramebuffer(n.DRAW_FRAMEBUFFER, [_]);
      }
    }
  }
  function ae(E) {
    return Math.min(s.maxSamples, E.samples);
  }
  function j(E) {
    const _ = i.get(E);
    return E.samples > 0 && e.has("WEBGL_multisampled_render_to_texture") === !0 && _.__useRenderToTexture !== !1;
  }
  function ee(E) {
    const _ = o.render.frame;
    u.get(E) !== _ && (u.set(E, _), E.update());
  }
  function te(E, _) {
    const I = E.colorSpace, k = E.format, Q = E.type;
    return E.isCompressedTexture === !0 || E.isVideoTexture === !0 || I !== Bs && I !== vi && (et.getTransfer(I) === ot ? (k !== xn || Q !== Bn) && console.warn("THREE.WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType.") : console.error("THREE.WebGLTextures: Unsupported texture color space:", I)), _;
  }
  function xe(E) {
    return typeof HTMLImageElement < "u" && E instanceof HTMLImageElement ? (c.width = E.naturalWidth || E.width, c.height = E.naturalHeight || E.height) : typeof VideoFrame < "u" && E instanceof VideoFrame ? (c.width = E.displayWidth, c.height = E.displayHeight) : (c.width = E.width, c.height = E.height), c;
  }
  this.allocateTextureUnit = V, this.resetTextureUnits = L, this.setTexture2D = ne, this.setTexture2DArray = J, this.setTexture3D = ie, this.setTextureCube = H, this.rebindTextures = R, this.setupRenderTarget = g, this.updateRenderTargetMipmap = W, this.updateMultisampleRenderTarget = z, this.setupDepthRenderbuffer = Ze, this.setupFrameBufferTexture = be, this.useMultisampledRTT = j;
}
function $S(n, e) {
  function t(i, s = vi) {
    let r;
    const o = et.getTransfer(s);
    if (i === Bn) return n.UNSIGNED_BYTE;
    if (i === Sc) return n.UNSIGNED_SHORT_4_4_4_4;
    if (i === yc) return n.UNSIGNED_SHORT_5_5_5_1;
    if (i === od) return n.UNSIGNED_INT_5_9_9_9_REV;
    if (i === ad) return n.UNSIGNED_INT_10F_11F_11F_REV;
    if (i === sd) return n.BYTE;
    if (i === rd) return n.SHORT;
    if (i === br) return n.UNSIGNED_SHORT;
    if (i === Mc) return n.INT;
    if (i === Yi) return n.UNSIGNED_INT;
    if (i === ei) return n.FLOAT;
    if (i === Ir) return n.HALF_FLOAT;
    if (i === ld) return n.ALPHA;
    if (i === cd) return n.RGB;
    if (i === xn) return n.RGBA;
    if (i === wr) return n.DEPTH_COMPONENT;
    if (i === Rr) return n.DEPTH_STENCIL;
    if (i === ud) return n.RED;
    if (i === Ec) return n.RED_INTEGER;
    if (i === hd) return n.RG;
    if (i === Tc) return n.RG_INTEGER;
    if (i === bc) return n.RGBA_INTEGER;
    if (i === So || i === yo || i === Eo || i === To)
      if (o === ot)
        if (r = e.get("WEBGL_compressed_texture_s3tc_srgb"), r !== null) {
          if (i === So) return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;
          if (i === yo) return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
          if (i === Eo) return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
          if (i === To) return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
        } else
          return null;
      else if (r = e.get("WEBGL_compressed_texture_s3tc"), r !== null) {
        if (i === So) return r.COMPRESSED_RGB_S3TC_DXT1_EXT;
        if (i === yo) return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;
        if (i === Eo) return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;
        if (i === To) return r.COMPRESSED_RGBA_S3TC_DXT5_EXT;
      } else
        return null;
    if (i === bl || i === Al || i === wl || i === Rl)
      if (r = e.get("WEBGL_compressed_texture_pvrtc"), r !== null) {
        if (i === bl) return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;
        if (i === Al) return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;
        if (i === wl) return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;
        if (i === Rl) return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
      } else
        return null;
    if (i === Cl || i === Pl || i === Dl)
      if (r = e.get("WEBGL_compressed_texture_etc"), r !== null) {
        if (i === Cl || i === Pl) return o === ot ? r.COMPRESSED_SRGB8_ETC2 : r.COMPRESSED_RGB8_ETC2;
        if (i === Dl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC : r.COMPRESSED_RGBA8_ETC2_EAC;
      } else
        return null;
    if (i === Ll || i === Il || i === Ul || i === Nl || i === Fl || i === Ol || i === Bl || i === zl || i === Hl || i === Vl || i === kl || i === Gl || i === Wl || i === Xl)
      if (r = e.get("WEBGL_compressed_texture_astc"), r !== null) {
        if (i === Ll) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR : r.COMPRESSED_RGBA_ASTC_4x4_KHR;
        if (i === Il) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR : r.COMPRESSED_RGBA_ASTC_5x4_KHR;
        if (i === Ul) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR : r.COMPRESSED_RGBA_ASTC_5x5_KHR;
        if (i === Nl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR : r.COMPRESSED_RGBA_ASTC_6x5_KHR;
        if (i === Fl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR : r.COMPRESSED_RGBA_ASTC_6x6_KHR;
        if (i === Ol) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR : r.COMPRESSED_RGBA_ASTC_8x5_KHR;
        if (i === Bl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR : r.COMPRESSED_RGBA_ASTC_8x6_KHR;
        if (i === zl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR : r.COMPRESSED_RGBA_ASTC_8x8_KHR;
        if (i === Hl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR : r.COMPRESSED_RGBA_ASTC_10x5_KHR;
        if (i === Vl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR : r.COMPRESSED_RGBA_ASTC_10x6_KHR;
        if (i === kl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR : r.COMPRESSED_RGBA_ASTC_10x8_KHR;
        if (i === Gl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR : r.COMPRESSED_RGBA_ASTC_10x10_KHR;
        if (i === Wl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR : r.COMPRESSED_RGBA_ASTC_12x10_KHR;
        if (i === Xl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR : r.COMPRESSED_RGBA_ASTC_12x12_KHR;
      } else
        return null;
    if (i === Yl || i === ql || i === jl)
      if (r = e.get("EXT_texture_compression_bptc"), r !== null) {
        if (i === Yl) return o === ot ? r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT : r.COMPRESSED_RGBA_BPTC_UNORM_EXT;
        if (i === ql) return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;
        if (i === jl) return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT;
      } else
        return null;
    if (i === Kl || i === $l || i === Zl || i === Jl)
      if (r = e.get("EXT_texture_compression_rgtc"), r !== null) {
        if (i === Kl) return r.COMPRESSED_RED_RGTC1_EXT;
        if (i === $l) return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;
        if (i === Zl) return r.COMPRESSED_RED_GREEN_RGTC2_EXT;
        if (i === Jl) return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
      } else
        return null;
    return i === Ar ? n.UNSIGNED_INT_24_8 : n[i] !== void 0 ? n[i] : null;
  }
  return { convert: t };
}
const ZS = `
void main() {

	gl_Position = vec4( position, 1.0 );

}`, JS = `
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;
class QS {
  /**
   * Constructs a new depth sensing module.
   */
  constructor() {
    this.texture = null, this.mesh = null, this.depthNear = 0, this.depthFar = 0;
  }
  /**
   * Inits the depth sensing module
   *
   * @param {XRWebGLDepthInformation} depthData - The XR depth data.
   * @param {XRRenderState} renderState - The XR render state.
   */
  init(e, t) {
    if (this.texture === null) {
      const i = new Ad(e.texture);
      (e.depthNear !== t.depthNear || e.depthFar !== t.depthFar) && (this.depthNear = e.depthNear, this.depthFar = e.depthFar), this.texture = i;
    }
  }
  /**
   * Returns a plane mesh that visualizes the depth texture.
   *
   * @param {ArrayCamera} cameraXR - The XR camera.
   * @return {?Mesh} The plane mesh.
   */
  getMesh(e) {
    if (this.texture !== null && this.mesh === null) {
      const t = e.cameras[0].viewport, i = new Ei({
        vertexShader: ZS,
        fragmentShader: JS,
        uniforms: {
          depthColor: { value: this.texture },
          depthWidth: { value: t.z },
          depthHeight: { value: t.w }
        }
      });
      this.mesh = new vt(new Hs(20, 20), i);
    }
    return this.mesh;
  }
  /**
   * Resets the module
   */
  reset() {
    this.texture = null, this.mesh = null;
  }
  /**
   * Returns a texture representing the depth of the user's environment.
   *
   * @return {?ExternalTexture} The depth texture.
   */
  getDepthTexture() {
    return this.texture;
  }
}
class ey extends Ji {
  /**
   * Constructs a new WebGL renderer.
   *
   * @param {WebGLRenderer} renderer - The renderer.
   * @param {WebGL2RenderingContext} gl - The rendering context.
   */
  constructor(e, t) {
    super();
    const i = this;
    let s = null, r = 1, o = null, a = "local-floor", l = 1, c = null, u = null, h = null, f = null, p = null, v = null;
    const x = typeof XRWebGLBinding < "u", m = new QS(), d = {}, b = t.getContextAttributes();
    let A = null, M = null;
    const C = [], w = [], P = new Ve();
    let U = null;
    const S = new rn();
    S.viewport = new lt();
    const y = new rn();
    y.viewport = new lt();
    const D = [S, y], L = new x0();
    let V = null, Z = null;
    this.cameraAutoUpdate = !0, this.enabled = !1, this.isPresenting = !1, this.getController = function(X) {
      let re = C[X];
      return re === void 0 && (re = new Va(), C[X] = re), re.getTargetRaySpace();
    }, this.getControllerGrip = function(X) {
      let re = C[X];
      return re === void 0 && (re = new Va(), C[X] = re), re.getGripSpace();
    }, this.getHand = function(X) {
      let re = C[X];
      return re === void 0 && (re = new Va(), C[X] = re), re.getHandSpace();
    };
    function ne(X) {
      const re = w.indexOf(X.inputSource);
      if (re === -1)
        return;
      const be = C[re];
      be !== void 0 && (be.update(X.inputSource, X.frame, c || o), be.dispatchEvent({ type: X.type, data: X.inputSource }));
    }
    function J() {
      s.removeEventListener("select", ne), s.removeEventListener("selectstart", ne), s.removeEventListener("selectend", ne), s.removeEventListener("squeeze", ne), s.removeEventListener("squeezestart", ne), s.removeEventListener("squeezeend", ne), s.removeEventListener("end", J), s.removeEventListener("inputsourceschange", ie);
      for (let X = 0; X < C.length; X++) {
        const re = w[X];
        re !== null && (w[X] = null, C[X].disconnect(re));
      }
      V = null, Z = null, m.reset();
      for (const X in d)
        delete d[X];
      e.setRenderTarget(A), p = null, f = null, h = null, s = null, M = null, Ae.stop(), i.isPresenting = !1, e.setPixelRatio(U), e.setSize(P.width, P.height, !1), i.dispatchEvent({ type: "sessionend" });
    }
    this.setFramebufferScaleFactor = function(X) {
      r = X, i.isPresenting === !0 && console.warn("THREE.WebXRManager: Cannot change framebuffer scale while presenting.");
    }, this.setReferenceSpaceType = function(X) {
      a = X, i.isPresenting === !0 && console.warn("THREE.WebXRManager: Cannot change reference space type while presenting.");
    }, this.getReferenceSpace = function() {
      return c || o;
    }, this.setReferenceSpace = function(X) {
      c = X;
    }, this.getBaseLayer = function() {
      return f !== null ? f : p;
    }, this.getBinding = function() {
      return h === null && x && (h = new XRWebGLBinding(s, t)), h;
    }, this.getFrame = function() {
      return v;
    }, this.getSession = function() {
      return s;
    }, this.setSession = async function(X) {
      if (s = X, s !== null) {
        if (A = e.getRenderTarget(), s.addEventListener("select", ne), s.addEventListener("selectstart", ne), s.addEventListener("selectend", ne), s.addEventListener("squeeze", ne), s.addEventListener("squeezestart", ne), s.addEventListener("squeezeend", ne), s.addEventListener("end", J), s.addEventListener("inputsourceschange", ie), b.xrCompatible !== !0 && await t.makeXRCompatible(), U = e.getPixelRatio(), e.getSize(P), x && "createProjectionLayer" in XRWebGLBinding.prototype) {
          let be = null, Be = null, Pe = null;
          b.depth && (Pe = b.stencil ? t.DEPTH24_STENCIL8 : t.DEPTH_COMPONENT24, be = b.stencil ? Rr : wr, Be = b.stencil ? Ar : Yi);
          const Ze = {
            colorFormat: t.RGBA8,
            depthFormat: Pe,
            scaleFactor: r
          };
          h = this.getBinding(), f = h.createProjectionLayer(Ze), s.updateRenderState({ layers: [f] }), e.setPixelRatio(1), e.setSize(f.textureWidth, f.textureHeight, !1), M = new ji(
            f.textureWidth,
            f.textureHeight,
            {
              format: xn,
              type: Bn,
              depthTexture: new bd(f.textureWidth, f.textureHeight, Be, void 0, void 0, void 0, void 0, void 0, void 0, be),
              stencilBuffer: b.stencil,
              colorSpace: e.outputColorSpace,
              samples: b.antialias ? 4 : 0,
              resolveDepthBuffer: f.ignoreDepthValues === !1,
              resolveStencilBuffer: f.ignoreDepthValues === !1
            }
          );
        } else {
          const be = {
            antialias: b.antialias,
            alpha: !0,
            depth: b.depth,
            stencil: b.stencil,
            framebufferScaleFactor: r
          };
          p = new XRWebGLLayer(s, t, be), s.updateRenderState({ baseLayer: p }), e.setPixelRatio(1), e.setSize(p.framebufferWidth, p.framebufferHeight, !1), M = new ji(
            p.framebufferWidth,
            p.framebufferHeight,
            {
              format: xn,
              type: Bn,
              colorSpace: e.outputColorSpace,
              stencilBuffer: b.stencil,
              resolveDepthBuffer: p.ignoreDepthValues === !1,
              resolveStencilBuffer: p.ignoreDepthValues === !1
            }
          );
        }
        M.isXRRenderTarget = !0, this.setFoveation(l), c = null, o = await s.requestReferenceSpace(a), Ae.setContext(s), Ae.start(), i.isPresenting = !0, i.dispatchEvent({ type: "sessionstart" });
      }
    }, this.getEnvironmentBlendMode = function() {
      if (s !== null)
        return s.environmentBlendMode;
    }, this.getDepthTexture = function() {
      return m.getDepthTexture();
    };
    function ie(X) {
      for (let re = 0; re < X.removed.length; re++) {
        const be = X.removed[re], Be = w.indexOf(be);
        Be >= 0 && (w[Be] = null, C[Be].disconnect(be));
      }
      for (let re = 0; re < X.added.length; re++) {
        const be = X.added[re];
        let Be = w.indexOf(be);
        if (Be === -1) {
          for (let Ze = 0; Ze < C.length; Ze++)
            if (Ze >= w.length) {
              w.push(be), Be = Ze;
              break;
            } else if (w[Ze] === null) {
              w[Ze] = be, Be = Ze;
              break;
            }
          if (Be === -1) break;
        }
        const Pe = C[Be];
        Pe && Pe.connect(be);
      }
    }
    const H = new N(), fe = new N();
    function ge(X, re, be) {
      H.setFromMatrixPosition(re.matrixWorld), fe.setFromMatrixPosition(be.matrixWorld);
      const Be = H.distanceTo(fe), Pe = re.projectionMatrix.elements, Ze = be.projectionMatrix.elements, R = Pe[14] / (Pe[10] - 1), g = Pe[14] / (Pe[10] + 1), W = (Pe[9] + 1) / Pe[5], K = (Pe[9] - 1) / Pe[5], Y = (Pe[8] - 1) / Pe[0], z = (Ze[8] + 1) / Ze[0], ae = R * Y, j = R * z, ee = Be / (-Y + z), te = ee * -Y;
      if (re.matrixWorld.decompose(X.position, X.quaternion, X.scale), X.translateX(te), X.translateZ(ee), X.matrixWorld.compose(X.position, X.quaternion, X.scale), X.matrixWorldInverse.copy(X.matrixWorld).invert(), Pe[10] === -1)
        X.projectionMatrix.copy(re.projectionMatrix), X.projectionMatrixInverse.copy(re.projectionMatrixInverse);
      else {
        const xe = R + ee, E = g + ee, _ = ae - te, I = j + (Be - te), k = W * g / E * xe, Q = K * g / E * xe;
        X.projectionMatrix.makePerspective(_, I, k, Q, xe, E), X.projectionMatrixInverse.copy(X.projectionMatrix).invert();
      }
    }
    function ye(X, re) {
      re === null ? X.matrixWorld.copy(X.matrix) : X.matrixWorld.multiplyMatrices(re.matrixWorld, X.matrix), X.matrixWorldInverse.copy(X.matrixWorld).invert();
    }
    this.updateCamera = function(X) {
      if (s === null) return;
      let re = X.near, be = X.far;
      m.texture !== null && (m.depthNear > 0 && (re = m.depthNear), m.depthFar > 0 && (be = m.depthFar)), L.near = y.near = S.near = re, L.far = y.far = S.far = be, (V !== L.near || Z !== L.far) && (s.updateRenderState({
        depthNear: L.near,
        depthFar: L.far
      }), V = L.near, Z = L.far), L.layers.mask = X.layers.mask | 6, S.layers.mask = L.layers.mask & 3, y.layers.mask = L.layers.mask & 5;
      const Be = X.parent, Pe = L.cameras;
      ye(L, Be);
      for (let Ze = 0; Ze < Pe.length; Ze++)
        ye(Pe[Ze], Be);
      Pe.length === 2 ? ge(L, S, y) : L.projectionMatrix.copy(S.projectionMatrix), Fe(X, L, Be);
    };
    function Fe(X, re, be) {
      be === null ? X.matrix.copy(re.matrixWorld) : (X.matrix.copy(be.matrixWorld), X.matrix.invert(), X.matrix.multiply(re.matrixWorld)), X.matrix.decompose(X.position, X.quaternion, X.scale), X.updateMatrixWorld(!0), X.projectionMatrix.copy(re.projectionMatrix), X.projectionMatrixInverse.copy(re.projectionMatrixInverse), X.isPerspectiveCamera && (X.fov = Ql * 2 * Math.atan(1 / X.projectionMatrix.elements[5]), X.zoom = 1);
    }
    this.getCamera = function() {
      return L;
    }, this.getFoveation = function() {
      if (!(f === null && p === null))
        return l;
    }, this.setFoveation = function(X) {
      l = X, f !== null && (f.fixedFoveation = X), p !== null && p.fixedFoveation !== void 0 && (p.fixedFoveation = X);
    }, this.hasDepthSensing = function() {
      return m.texture !== null;
    }, this.getDepthSensingMesh = function() {
      return m.getMesh(L);
    }, this.getCameraTexture = function(X) {
      return d[X];
    };
    let Je = null;
    function Ge(X, re) {
      if (u = re.getViewerPose(c || o), v = re, u !== null) {
        const be = u.views;
        p !== null && (e.setRenderTargetFramebuffer(M, p.framebuffer), e.setRenderTarget(M));
        let Be = !1;
        be.length !== L.cameras.length && (L.cameras.length = 0, Be = !0);
        for (let g = 0; g < be.length; g++) {
          const W = be[g];
          let K = null;
          if (p !== null)
            K = p.getViewport(W);
          else {
            const z = h.getViewSubImage(f, W);
            K = z.viewport, g === 0 && (e.setRenderTargetTextures(
              M,
              z.colorTexture,
              z.depthStencilTexture
            ), e.setRenderTarget(M));
          }
          let Y = D[g];
          Y === void 0 && (Y = new rn(), Y.layers.enable(g), Y.viewport = new lt(), D[g] = Y), Y.matrix.fromArray(W.transform.matrix), Y.matrix.decompose(Y.position, Y.quaternion, Y.scale), Y.projectionMatrix.fromArray(W.projectionMatrix), Y.projectionMatrixInverse.copy(Y.projectionMatrix).invert(), Y.viewport.set(K.x, K.y, K.width, K.height), g === 0 && (L.matrix.copy(Y.matrix), L.matrix.decompose(L.position, L.quaternion, L.scale)), Be === !0 && L.cameras.push(Y);
        }
        const Pe = s.enabledFeatures;
        if (Pe && Pe.includes("depth-sensing") && s.depthUsage == "gpu-optimized" && x) {
          h = i.getBinding();
          const g = h.getDepthInformation(be[0]);
          g && g.isValid && g.texture && m.init(g, s.renderState);
        }
        if (Pe && Pe.includes("camera-access") && x) {
          e.state.unbindTexture(), h = i.getBinding();
          for (let g = 0; g < be.length; g++) {
            const W = be[g].camera;
            if (W) {
              let K = d[W];
              K || (K = new Ad(), d[W] = K);
              const Y = h.getCameraImage(W);
              K.sourceTexture = Y;
            }
          }
        }
      }
      for (let be = 0; be < C.length; be++) {
        const Be = w[be], Pe = C[be];
        Be !== null && Pe !== void 0 && Pe.update(Be, re, c || o);
      }
      Je && Je(X, re), re.detectedPlanes && i.dispatchEvent({ type: "planesdetected", data: re }), v = null;
    }
    const Ae = new Cd();
    Ae.setAnimationLoop(Ge), this.setAnimationLoop = function(X) {
      Je = X;
    }, this.dispose = function() {
    };
  }
}
const Ni = /* @__PURE__ */ new zn(), ty = /* @__PURE__ */ new pt();
function ny(n, e) {
  function t(m, d) {
    m.matrixAutoUpdate === !0 && m.updateMatrix(), d.value.copy(m.matrix);
  }
  function i(m, d) {
    d.color.getRGB(m.fogColor.value, Md(n)), d.isFog ? (m.fogNear.value = d.near, m.fogFar.value = d.far) : d.isFogExp2 && (m.fogDensity.value = d.density);
  }
  function s(m, d, b, A, M) {
    d.isMeshBasicMaterial || d.isMeshLambertMaterial ? r(m, d) : d.isMeshToonMaterial ? (r(m, d), h(m, d)) : d.isMeshPhongMaterial ? (r(m, d), u(m, d)) : d.isMeshStandardMaterial ? (r(m, d), f(m, d), d.isMeshPhysicalMaterial && p(m, d, M)) : d.isMeshMatcapMaterial ? (r(m, d), v(m, d)) : d.isMeshDepthMaterial ? r(m, d) : d.isMeshDistanceMaterial ? (r(m, d), x(m, d)) : d.isMeshNormalMaterial ? r(m, d) : d.isLineBasicMaterial ? (o(m, d), d.isLineDashedMaterial && a(m, d)) : d.isPointsMaterial ? l(m, d, b, A) : d.isSpriteMaterial ? c(m, d) : d.isShadowMaterial ? (m.color.value.copy(d.color), m.opacity.value = d.opacity) : d.isShaderMaterial && (d.uniformsNeedUpdate = !1);
  }
  function r(m, d) {
    m.opacity.value = d.opacity, d.color && m.diffuse.value.copy(d.color), d.emissive && m.emissive.value.copy(d.emissive).multiplyScalar(d.emissiveIntensity), d.map && (m.map.value = d.map, t(d.map, m.mapTransform)), d.alphaMap && (m.alphaMap.value = d.alphaMap, t(d.alphaMap, m.alphaMapTransform)), d.bumpMap && (m.bumpMap.value = d.bumpMap, t(d.bumpMap, m.bumpMapTransform), m.bumpScale.value = d.bumpScale, d.side === Wt && (m.bumpScale.value *= -1)), d.normalMap && (m.normalMap.value = d.normalMap, t(d.normalMap, m.normalMapTransform), m.normalScale.value.copy(d.normalScale), d.side === Wt && m.normalScale.value.negate()), d.displacementMap && (m.displacementMap.value = d.displacementMap, t(d.displacementMap, m.displacementMapTransform), m.displacementScale.value = d.displacementScale, m.displacementBias.value = d.displacementBias), d.emissiveMap && (m.emissiveMap.value = d.emissiveMap, t(d.emissiveMap, m.emissiveMapTransform)), d.specularMap && (m.specularMap.value = d.specularMap, t(d.specularMap, m.specularMapTransform)), d.alphaTest > 0 && (m.alphaTest.value = d.alphaTest);
    const b = e.get(d), A = b.envMap, M = b.envMapRotation;
    A && (m.envMap.value = A, Ni.copy(M), Ni.x *= -1, Ni.y *= -1, Ni.z *= -1, A.isCubeTexture && A.isRenderTargetTexture === !1 && (Ni.y *= -1, Ni.z *= -1), m.envMapRotation.value.setFromMatrix4(ty.makeRotationFromEuler(Ni)), m.flipEnvMap.value = A.isCubeTexture && A.isRenderTargetTexture === !1 ? -1 : 1, m.reflectivity.value = d.reflectivity, m.ior.value = d.ior, m.refractionRatio.value = d.refractionRatio), d.lightMap && (m.lightMap.value = d.lightMap, m.lightMapIntensity.value = d.lightMapIntensity, t(d.lightMap, m.lightMapTransform)), d.aoMap && (m.aoMap.value = d.aoMap, m.aoMapIntensity.value = d.aoMapIntensity, t(d.aoMap, m.aoMapTransform));
  }
  function o(m, d) {
    m.diffuse.value.copy(d.color), m.opacity.value = d.opacity, d.map && (m.map.value = d.map, t(d.map, m.mapTransform));
  }
  function a(m, d) {
    m.dashSize.value = d.dashSize, m.totalSize.value = d.dashSize + d.gapSize, m.scale.value = d.scale;
  }
  function l(m, d, b, A) {
    m.diffuse.value.copy(d.color), m.opacity.value = d.opacity, m.size.value = d.size * b, m.scale.value = A * 0.5, d.map && (m.map.value = d.map, t(d.map, m.uvTransform)), d.alphaMap && (m.alphaMap.value = d.alphaMap, t(d.alphaMap, m.alphaMapTransform)), d.alphaTest > 0 && (m.alphaTest.value = d.alphaTest);
  }
  function c(m, d) {
    m.diffuse.value.copy(d.color), m.opacity.value = d.opacity, m.rotation.value = d.rotation, d.map && (m.map.value = d.map, t(d.map, m.mapTransform)), d.alphaMap && (m.alphaMap.value = d.alphaMap, t(d.alphaMap, m.alphaMapTransform)), d.alphaTest > 0 && (m.alphaTest.value = d.alphaTest);
  }
  function u(m, d) {
    m.specular.value.copy(d.specular), m.shininess.value = Math.max(d.shininess, 1e-4);
  }
  function h(m, d) {
    d.gradientMap && (m.gradientMap.value = d.gradientMap);
  }
  function f(m, d) {
    m.metalness.value = d.metalness, d.metalnessMap && (m.metalnessMap.value = d.metalnessMap, t(d.metalnessMap, m.metalnessMapTransform)), m.roughness.value = d.roughness, d.roughnessMap && (m.roughnessMap.value = d.roughnessMap, t(d.roughnessMap, m.roughnessMapTransform)), d.envMap && (m.envMapIntensity.value = d.envMapIntensity);
  }
  function p(m, d, b) {
    m.ior.value = d.ior, d.sheen > 0 && (m.sheenColor.value.copy(d.sheenColor).multiplyScalar(d.sheen), m.sheenRoughness.value = d.sheenRoughness, d.sheenColorMap && (m.sheenColorMap.value = d.sheenColorMap, t(d.sheenColorMap, m.sheenColorMapTransform)), d.sheenRoughnessMap && (m.sheenRoughnessMap.value = d.sheenRoughnessMap, t(d.sheenRoughnessMap, m.sheenRoughnessMapTransform))), d.clearcoat > 0 && (m.clearcoat.value = d.clearcoat, m.clearcoatRoughness.value = d.clearcoatRoughness, d.clearcoatMap && (m.clearcoatMap.value = d.clearcoatMap, t(d.clearcoatMap, m.clearcoatMapTransform)), d.clearcoatRoughnessMap && (m.clearcoatRoughnessMap.value = d.clearcoatRoughnessMap, t(d.clearcoatRoughnessMap, m.clearcoatRoughnessMapTransform)), d.clearcoatNormalMap && (m.clearcoatNormalMap.value = d.clearcoatNormalMap, t(d.clearcoatNormalMap, m.clearcoatNormalMapTransform), m.clearcoatNormalScale.value.copy(d.clearcoatNormalScale), d.side === Wt && m.clearcoatNormalScale.value.negate())), d.dispersion > 0 && (m.dispersion.value = d.dispersion), d.iridescence > 0 && (m.iridescence.value = d.iridescence, m.iridescenceIOR.value = d.iridescenceIOR, m.iridescenceThicknessMinimum.value = d.iridescenceThicknessRange[0], m.iridescenceThicknessMaximum.value = d.iridescenceThicknessRange[1], d.iridescenceMap && (m.iridescenceMap.value = d.iridescenceMap, t(d.iridescenceMap, m.iridescenceMapTransform)), d.iridescenceThicknessMap && (m.iridescenceThicknessMap.value = d.iridescenceThicknessMap, t(d.iridescenceThicknessMap, m.iridescenceThicknessMapTransform))), d.transmission > 0 && (m.transmission.value = d.transmission, m.transmissionSamplerMap.value = b.texture, m.transmissionSamplerSize.value.set(b.width, b.height), d.transmissionMap && (m.transmissionMap.value = d.transmissionMap, t(d.transmissionMap, m.transmissionMapTransform)), m.thickness.value = d.thickness, d.thicknessMap && (m.thicknessMap.value = d.thicknessMap, t(d.thicknessMap, m.thicknessMapTransform)), m.attenuationDistance.value = d.attenuationDistance, m.attenuationColor.value.copy(d.attenuationColor)), d.anisotropy > 0 && (m.anisotropyVector.value.set(d.anisotropy * Math.cos(d.anisotropyRotation), d.anisotropy * Math.sin(d.anisotropyRotation)), d.anisotropyMap && (m.anisotropyMap.value = d.anisotropyMap, t(d.anisotropyMap, m.anisotropyMapTransform))), m.specularIntensity.value = d.specularIntensity, m.specularColor.value.copy(d.specularColor), d.specularColorMap && (m.specularColorMap.value = d.specularColorMap, t(d.specularColorMap, m.specularColorMapTransform)), d.specularIntensityMap && (m.specularIntensityMap.value = d.specularIntensityMap, t(d.specularIntensityMap, m.specularIntensityMapTransform));
  }
  function v(m, d) {
    d.matcap && (m.matcap.value = d.matcap);
  }
  function x(m, d) {
    const b = e.get(d).light;
    m.referencePosition.value.setFromMatrixPosition(b.matrixWorld), m.nearDistance.value = b.shadow.camera.near, m.farDistance.value = b.shadow.camera.far;
  }
  return {
    refreshFogUniforms: i,
    refreshMaterialUniforms: s
  };
}
function iy(n, e, t, i) {
  let s = {}, r = {}, o = [];
  const a = n.getParameter(n.MAX_UNIFORM_BUFFER_BINDINGS);
  function l(b, A) {
    const M = A.program;
    i.uniformBlockBinding(b, M);
  }
  function c(b, A) {
    let M = s[b.id];
    M === void 0 && (v(b), M = u(b), s[b.id] = M, b.addEventListener("dispose", m));
    const C = A.program;
    i.updateUBOMapping(b, C);
    const w = e.render.frame;
    r[b.id] !== w && (f(b), r[b.id] = w);
  }
  function u(b) {
    const A = h();
    b.__bindingPointIndex = A;
    const M = n.createBuffer(), C = b.__size, w = b.usage;
    return n.bindBuffer(n.UNIFORM_BUFFER, M), n.bufferData(n.UNIFORM_BUFFER, C, w), n.bindBuffer(n.UNIFORM_BUFFER, null), n.bindBufferBase(n.UNIFORM_BUFFER, A, M), M;
  }
  function h() {
    for (let b = 0; b < a; b++)
      if (o.indexOf(b) === -1)
        return o.push(b), b;
    return console.error("THREE.WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."), 0;
  }
  function f(b) {
    const A = s[b.id], M = b.uniforms, C = b.__cache;
    n.bindBuffer(n.UNIFORM_BUFFER, A);
    for (let w = 0, P = M.length; w < P; w++) {
      const U = Array.isArray(M[w]) ? M[w] : [M[w]];
      for (let S = 0, y = U.length; S < y; S++) {
        const D = U[S];
        if (p(D, w, S, C) === !0) {
          const L = D.__offset, V = Array.isArray(D.value) ? D.value : [D.value];
          let Z = 0;
          for (let ne = 0; ne < V.length; ne++) {
            const J = V[ne], ie = x(J);
            typeof J == "number" || typeof J == "boolean" ? (D.__data[0] = J, n.bufferSubData(n.UNIFORM_BUFFER, L + Z, D.__data)) : J.isMatrix3 ? (D.__data[0] = J.elements[0], D.__data[1] = J.elements[1], D.__data[2] = J.elements[2], D.__data[3] = 0, D.__data[4] = J.elements[3], D.__data[5] = J.elements[4], D.__data[6] = J.elements[5], D.__data[7] = 0, D.__data[8] = J.elements[6], D.__data[9] = J.elements[7], D.__data[10] = J.elements[8], D.__data[11] = 0) : (J.toArray(D.__data, Z), Z += ie.storage / Float32Array.BYTES_PER_ELEMENT);
          }
          n.bufferSubData(n.UNIFORM_BUFFER, L, D.__data);
        }
      }
    }
    n.bindBuffer(n.UNIFORM_BUFFER, null);
  }
  function p(b, A, M, C) {
    const w = b.value, P = A + "_" + M;
    if (C[P] === void 0)
      return typeof w == "number" || typeof w == "boolean" ? C[P] = w : C[P] = w.clone(), !0;
    {
      const U = C[P];
      if (typeof w == "number" || typeof w == "boolean") {
        if (U !== w)
          return C[P] = w, !0;
      } else if (U.equals(w) === !1)
        return U.copy(w), !0;
    }
    return !1;
  }
  function v(b) {
    const A = b.uniforms;
    let M = 0;
    const C = 16;
    for (let P = 0, U = A.length; P < U; P++) {
      const S = Array.isArray(A[P]) ? A[P] : [A[P]];
      for (let y = 0, D = S.length; y < D; y++) {
        const L = S[y], V = Array.isArray(L.value) ? L.value : [L.value];
        for (let Z = 0, ne = V.length; Z < ne; Z++) {
          const J = V[Z], ie = x(J), H = M % C, fe = H % ie.boundary, ge = H + fe;
          M += fe, ge !== 0 && C - ge < ie.storage && (M += C - ge), L.__data = new Float32Array(ie.storage / Float32Array.BYTES_PER_ELEMENT), L.__offset = M, M += ie.storage;
        }
      }
    }
    const w = M % C;
    return w > 0 && (M += C - w), b.__size = M, b.__cache = {}, this;
  }
  function x(b) {
    const A = {
      boundary: 0,
      // bytes
      storage: 0
      // bytes
    };
    return typeof b == "number" || typeof b == "boolean" ? (A.boundary = 4, A.storage = 4) : b.isVector2 ? (A.boundary = 8, A.storage = 8) : b.isVector3 || b.isColor ? (A.boundary = 16, A.storage = 12) : b.isVector4 ? (A.boundary = 16, A.storage = 16) : b.isMatrix3 ? (A.boundary = 48, A.storage = 48) : b.isMatrix4 ? (A.boundary = 64, A.storage = 64) : b.isTexture ? console.warn("THREE.WebGLRenderer: Texture samplers can not be part of an uniforms group.") : console.warn("THREE.WebGLRenderer: Unsupported uniform value type.", b), A;
  }
  function m(b) {
    const A = b.target;
    A.removeEventListener("dispose", m);
    const M = o.indexOf(A.__bindingPointIndex);
    o.splice(M, 1), n.deleteBuffer(s[A.id]), delete s[A.id], delete r[A.id];
  }
  function d() {
    for (const b in s)
      n.deleteBuffer(s[b]);
    o = [], s = {}, r = {};
  }
  return {
    bind: l,
    update: c,
    dispose: d
  };
}
class sy {
  /**
   * Constructs a new WebGL renderer.
   *
   * @param {WebGLRenderer~Options} [parameters] - The configuration parameter.
   */
  constructor(e = {}) {
    const {
      canvas: t = Ig(),
      context: i = null,
      depth: s = !0,
      stencil: r = !1,
      alpha: o = !1,
      antialias: a = !1,
      premultipliedAlpha: l = !0,
      preserveDrawingBuffer: c = !1,
      powerPreference: u = "default",
      failIfMajorPerformanceCaveat: h = !1,
      reversedDepthBuffer: f = !1
    } = e;
    this.isWebGLRenderer = !0;
    let p;
    if (i !== null) {
      if (typeof WebGLRenderingContext < "u" && i instanceof WebGLRenderingContext)
        throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");
      p = i.getContextAttributes().alpha;
    } else
      p = o;
    const v = new Uint32Array(4), x = new Int32Array(4);
    let m = null, d = null;
    const b = [], A = [];
    this.domElement = t, this.debug = {
      /**
       * Enables error checking and reporting when shader programs are being compiled.
       * @type {boolean}
       */
      checkShaderErrors: !0,
      /**
       * Callback for custom error reporting.
       * @type {?Function}
       */
      onShaderError: null
    }, this.autoClear = !0, this.autoClearColor = !0, this.autoClearDepth = !0, this.autoClearStencil = !0, this.sortObjects = !0, this.clippingPlanes = [], this.localClippingEnabled = !1, this.toneMapping = Mi, this.toneMappingExposure = 1, this.transmissionResolutionScale = 1;
    const M = this;
    let C = !1;
    this._outputColorSpace = sn;
    let w = 0, P = 0, U = null, S = -1, y = null;
    const D = new lt(), L = new lt();
    let V = null;
    const Z = new Xe(0);
    let ne = 0, J = t.width, ie = t.height, H = 1, fe = null, ge = null;
    const ye = new lt(0, 0, J, ie), Fe = new lt(0, 0, J, ie);
    let Je = !1;
    const Ge = new wc();
    let Ae = !1, X = !1;
    const re = new pt(), be = new N(), Be = new lt(), Pe = { background: null, fog: null, environment: null, overrideMaterial: null, isScene: !0 };
    let Ze = !1;
    function R() {
      return U === null ? H : 1;
    }
    let g = i;
    function W(T, O) {
      return t.getContext(T, O);
    }
    try {
      const T = {
        alpha: !0,
        depth: s,
        stencil: r,
        antialias: a,
        premultipliedAlpha: l,
        preserveDrawingBuffer: c,
        powerPreference: u,
        failIfMajorPerformanceCaveat: h
      };
      if ("setAttribute" in t && t.setAttribute("data-engine", `three.js r${xc}`), t.addEventListener("webglcontextlost", de, !1), t.addEventListener("webglcontextrestored", Re, !1), t.addEventListener("webglcontextcreationerror", ce, !1), g === null) {
        const O = "webgl2";
        if (g = W(O, T), g === null)
          throw W(O) ? new Error("Error creating WebGL context with your selected attributes.") : new Error("Error creating WebGL context.");
      }
    } catch (T) {
      throw console.error("THREE.WebGLRenderer: " + T.message), T;
    }
    let K, Y, z, ae, j, ee, te, xe, E, _, I, k, Q, G, pe, oe, Se, Ee, le, ve, Ce, Te, me, ke;
    function F() {
      K = new pM(g), K.init(), Te = new $S(g, K), Y = new aM(g, K, e, Te), z = new jS(g, K), Y.reversedDepthBuffer && f && z.buffers.depth.setReversed(!0), ae = new gM(g), j = new NS(), ee = new KS(g, K, z, j, Y, Te, ae), te = new cM(M), xe = new dM(M), E = new E0(g), me = new rM(g, E), _ = new mM(g, E, ae, me), I = new xM(g, _, E, ae), le = new vM(g, Y, ee), oe = new lM(j), k = new US(M, te, xe, K, Y, me, oe), Q = new ny(M, j), G = new OS(), pe = new GS(K), Ee = new sM(M, te, xe, z, I, p, l), Se = new YS(M, I, Y), ke = new iy(g, ae, Y, z), ve = new oM(g, K, ae), Ce = new _M(g, K, ae), ae.programs = k.programs, M.capabilities = Y, M.extensions = K, M.properties = j, M.renderLists = G, M.shadowMap = Se, M.state = z, M.info = ae;
    }
    F();
    const he = new ey(M, g);
    this.xr = he, this.getContext = function() {
      return g;
    }, this.getContextAttributes = function() {
      return g.getContextAttributes();
    }, this.forceContextLoss = function() {
      const T = K.get("WEBGL_lose_context");
      T && T.loseContext();
    }, this.forceContextRestore = function() {
      const T = K.get("WEBGL_lose_context");
      T && T.restoreContext();
    }, this.getPixelRatio = function() {
      return H;
    }, this.setPixelRatio = function(T) {
      T !== void 0 && (H = T, this.setSize(J, ie, !1));
    }, this.getSize = function(T) {
      return T.set(J, ie);
    }, this.setSize = function(T, O, q = !0) {
      if (he.isPresenting) {
        console.warn("THREE.WebGLRenderer: Can't change size while VR device is presenting.");
        return;
      }
      J = T, ie = O, t.width = Math.floor(T * H), t.height = Math.floor(O * H), q === !0 && (t.style.width = T + "px", t.style.height = O + "px"), this.setViewport(0, 0, T, O);
    }, this.getDrawingBufferSize = function(T) {
      return T.set(J * H, ie * H).floor();
    }, this.setDrawingBufferSize = function(T, O, q) {
      J = T, ie = O, H = q, t.width = Math.floor(T * q), t.height = Math.floor(O * q), this.setViewport(0, 0, T, O);
    }, this.getCurrentViewport = function(T) {
      return T.copy(D);
    }, this.getViewport = function(T) {
      return T.copy(ye);
    }, this.setViewport = function(T, O, q, $) {
      T.isVector4 ? ye.set(T.x, T.y, T.z, T.w) : ye.set(T, O, q, $), z.viewport(D.copy(ye).multiplyScalar(H).round());
    }, this.getScissor = function(T) {
      return T.copy(Fe);
    }, this.setScissor = function(T, O, q, $) {
      T.isVector4 ? Fe.set(T.x, T.y, T.z, T.w) : Fe.set(T, O, q, $), z.scissor(L.copy(Fe).multiplyScalar(H).round());
    }, this.getScissorTest = function() {
      return Je;
    }, this.setScissorTest = function(T) {
      z.setScissorTest(Je = T);
    }, this.setOpaqueSort = function(T) {
      fe = T;
    }, this.setTransparentSort = function(T) {
      ge = T;
    }, this.getClearColor = function(T) {
      return T.copy(Ee.getClearColor());
    }, this.setClearColor = function() {
      Ee.setClearColor(...arguments);
    }, this.getClearAlpha = function() {
      return Ee.getClearAlpha();
    }, this.setClearAlpha = function() {
      Ee.setClearAlpha(...arguments);
    }, this.clear = function(T = !0, O = !0, q = !0) {
      let $ = 0;
      if (T) {
        let B = !1;
        if (U !== null) {
          const ue = U.texture.format;
          B = ue === bc || ue === Tc || ue === Ec;
        }
        if (B) {
          const ue = U.texture.type, Me = ue === Bn || ue === Yi || ue === br || ue === Ar || ue === Sc || ue === yc, De = Ee.getClearColor(), we = Ee.getClearAlpha(), Oe = De.r, He = De.g, Ie = De.b;
          Me ? (v[0] = Oe, v[1] = He, v[2] = Ie, v[3] = we, g.clearBufferuiv(g.COLOR, 0, v)) : (x[0] = Oe, x[1] = He, x[2] = Ie, x[3] = we, g.clearBufferiv(g.COLOR, 0, x));
        } else
          $ |= g.COLOR_BUFFER_BIT;
      }
      O && ($ |= g.DEPTH_BUFFER_BIT), q && ($ |= g.STENCIL_BUFFER_BIT, this.state.buffers.stencil.setMask(4294967295)), g.clear($);
    }, this.clearColor = function() {
      this.clear(!0, !1, !1);
    }, this.clearDepth = function() {
      this.clear(!1, !0, !1);
    }, this.clearStencil = function() {
      this.clear(!1, !1, !0);
    }, this.dispose = function() {
      t.removeEventListener("webglcontextlost", de, !1), t.removeEventListener("webglcontextrestored", Re, !1), t.removeEventListener("webglcontextcreationerror", ce, !1), Ee.dispose(), G.dispose(), pe.dispose(), j.dispose(), te.dispose(), xe.dispose(), I.dispose(), me.dispose(), ke.dispose(), k.dispose(), he.dispose(), he.removeEventListener("sessionstart", bn), he.removeEventListener("sessionend", Hc), Ti.stop();
    };
    function de(T) {
      T.preventDefault(), console.log("THREE.WebGLRenderer: Context Lost."), C = !0;
    }
    function Re() {
      console.log("THREE.WebGLRenderer: Context Restored."), C = !1;
      const T = ae.autoReset, O = Se.enabled, q = Se.autoUpdate, $ = Se.needsUpdate, B = Se.type;
      F(), ae.autoReset = T, Se.enabled = O, Se.autoUpdate = q, Se.needsUpdate = $, Se.type = B;
    }
    function ce(T) {
      console.error("THREE.WebGLRenderer: A WebGL context could not be created. Reason: ", T.statusMessage);
    }
    function se(T) {
      const O = T.target;
      O.removeEventListener("dispose", se), Le(O);
    }
    function Le(T) {
      We(T), j.remove(T);
    }
    function We(T) {
      const O = j.get(T).programs;
      O !== void 0 && (O.forEach(function(q) {
        k.releaseProgram(q);
      }), T.isShaderMaterial && k.releaseShaderCache(T));
    }
    this.renderBufferDirect = function(T, O, q, $, B, ue) {
      O === null && (O = Pe);
      const Me = B.isMesh && B.matrixWorld.determinant() < 0, De = Vd(T, O, q, $, B);
      z.setMaterial($, Me);
      let we = q.index, Oe = 1;
      if ($.wireframe === !0) {
        if (we = _.getWireframeAttribute(q), we === void 0) return;
        Oe = 2;
      }
      const He = q.drawRange, Ie = q.attributes.position;
      let $e = He.start * Oe, rt = (He.start + He.count) * Oe;
      ue !== null && ($e = Math.max($e, ue.start * Oe), rt = Math.min(rt, (ue.start + ue.count) * Oe)), we !== null ? ($e = Math.max($e, 0), rt = Math.min(rt, we.count)) : Ie != null && ($e = Math.max($e, 0), rt = Math.min(rt, Ie.count));
      const Mt = rt - $e;
      if (Mt < 0 || Mt === 1 / 0) return;
      me.setup(B, $, De, q, we);
      let dt, ct = ve;
      if (we !== null && (dt = E.get(we), ct = Ce, ct.setIndex(dt)), B.isMesh)
        $.wireframe === !0 ? (z.setLineWidth($.wireframeLinewidth * R()), ct.setMode(g.LINES)) : ct.setMode(g.TRIANGLES);
      else if (B.isLine) {
        let Ue = $.linewidth;
        Ue === void 0 && (Ue = 1), z.setLineWidth(Ue * R()), B.isLineSegments ? ct.setMode(g.LINES) : B.isLineLoop ? ct.setMode(g.LINE_LOOP) : ct.setMode(g.LINE_STRIP);
      } else B.isPoints ? ct.setMode(g.POINTS) : B.isSprite && ct.setMode(g.TRIANGLES);
      if (B.isBatchedMesh)
        if (B._multiDrawInstances !== null)
          Cr("THREE.WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."), ct.renderMultiDrawInstances(B._multiDrawStarts, B._multiDrawCounts, B._multiDrawCount, B._multiDrawInstances);
        else if (K.get("WEBGL_multi_draw"))
          ct.renderMultiDraw(B._multiDrawStarts, B._multiDrawCounts, B._multiDrawCount);
        else {
          const Ue = B._multiDrawStarts, _t = B._multiDrawCounts, Qe = B._multiDrawCount, Jt = we ? E.get(we).bytesPerElement : 1, es = j.get($).currentProgram.getUniforms();
          for (let Qt = 0; Qt < Qe; Qt++)
            es.setValue(g, "_gl_DrawID", Qt), ct.render(Ue[Qt] / Jt, _t[Qt]);
        }
      else if (B.isInstancedMesh)
        ct.renderInstances($e, Mt, B.count);
      else if (q.isInstancedBufferGeometry) {
        const Ue = q._maxInstanceCount !== void 0 ? q._maxInstanceCount : 1 / 0, _t = Math.min(q.instanceCount, Ue);
        ct.renderInstances($e, Mt, _t);
      } else
        ct.render($e, Mt);
    };
    function ht(T, O, q) {
      T.transparent === !0 && T.side === Qn && T.forceSinglePass === !1 ? (T.side = Wt, T.needsUpdate = !0, Br(T, O, q), T.side = yi, T.needsUpdate = !0, Br(T, O, q), T.side = Qn) : Br(T, O, q);
    }
    this.compile = function(T, O, q = null) {
      q === null && (q = T), d = pe.get(q), d.init(O), A.push(d), q.traverseVisible(function(B) {
        B.isLight && B.layers.test(O.layers) && (d.pushLight(B), B.castShadow && d.pushShadow(B));
      }), T !== q && T.traverseVisible(function(B) {
        B.isLight && B.layers.test(O.layers) && (d.pushLight(B), B.castShadow && d.pushShadow(B));
      }), d.setupLights();
      const $ = /* @__PURE__ */ new Set();
      return T.traverse(function(B) {
        if (!(B.isMesh || B.isPoints || B.isLine || B.isSprite))
          return;
        const ue = B.material;
        if (ue)
          if (Array.isArray(ue))
            for (let Me = 0; Me < ue.length; Me++) {
              const De = ue[Me];
              ht(De, q, B), $.add(De);
            }
          else
            ht(ue, q, B), $.add(ue);
      }), d = A.pop(), $;
    }, this.compileAsync = function(T, O, q = null) {
      const $ = this.compile(T, O, q);
      return new Promise((B) => {
        function ue() {
          if ($.forEach(function(Me) {
            j.get(Me).currentProgram.isReady() && $.delete(Me);
          }), $.size === 0) {
            B(T);
            return;
          }
          setTimeout(ue, 10);
        }
        K.get("KHR_parallel_shader_compile") !== null ? ue() : setTimeout(ue, 10);
      });
    };
    let nt = null;
    function Hn(T) {
      nt && nt(T);
    }
    function bn() {
      Ti.stop();
    }
    function Hc() {
      Ti.start();
    }
    const Ti = new Cd();
    Ti.setAnimationLoop(Hn), typeof self < "u" && Ti.setContext(self), this.setAnimationLoop = function(T) {
      nt = T, he.setAnimationLoop(T), T === null ? Ti.stop() : Ti.start();
    }, he.addEventListener("sessionstart", bn), he.addEventListener("sessionend", Hc), this.render = function(T, O) {
      if (O !== void 0 && O.isCamera !== !0) {
        console.error("THREE.WebGLRenderer.render: camera is not an instance of THREE.Camera.");
        return;
      }
      if (C === !0) return;
      if (T.matrixWorldAutoUpdate === !0 && T.updateMatrixWorld(), O.parent === null && O.matrixWorldAutoUpdate === !0 && O.updateMatrixWorld(), he.enabled === !0 && he.isPresenting === !0 && (he.cameraAutoUpdate === !0 && he.updateCamera(O), O = he.getCamera()), T.isScene === !0 && T.onBeforeRender(M, T, O, U), d = pe.get(T, A.length), d.init(O), A.push(d), re.multiplyMatrices(O.projectionMatrix, O.matrixWorldInverse), Ge.setFromProjectionMatrix(re, Nn, O.reversedDepth), X = this.localClippingEnabled, Ae = oe.init(this.clippingPlanes, X), m = G.get(T, b.length), m.init(), b.push(m), he.enabled === !0 && he.isPresenting === !0) {
        const ue = M.xr.getDepthSensingMesh();
        ue !== null && sa(ue, O, -1 / 0, M.sortObjects);
      }
      sa(T, O, 0, M.sortObjects), m.finish(), M.sortObjects === !0 && m.sort(fe, ge), Ze = he.enabled === !1 || he.isPresenting === !1 || he.hasDepthSensing() === !1, Ze && Ee.addToRenderList(m, T), this.info.render.frame++, Ae === !0 && oe.beginShadows();
      const q = d.state.shadowsArray;
      Se.render(q, T, O), Ae === !0 && oe.endShadows(), this.info.autoReset === !0 && this.info.reset();
      const $ = m.opaque, B = m.transmissive;
      if (d.setupLights(), O.isArrayCamera) {
        const ue = O.cameras;
        if (B.length > 0)
          for (let Me = 0, De = ue.length; Me < De; Me++) {
            const we = ue[Me];
            kc($, B, T, we);
          }
        Ze && Ee.render(T);
        for (let Me = 0, De = ue.length; Me < De; Me++) {
          const we = ue[Me];
          Vc(m, T, we, we.viewport);
        }
      } else
        B.length > 0 && kc($, B, T, O), Ze && Ee.render(T), Vc(m, T, O);
      U !== null && P === 0 && (ee.updateMultisampleRenderTarget(U), ee.updateRenderTargetMipmap(U)), T.isScene === !0 && T.onAfterRender(M, T, O), me.resetDefaultState(), S = -1, y = null, A.pop(), A.length > 0 ? (d = A[A.length - 1], Ae === !0 && oe.setGlobalState(M.clippingPlanes, d.state.camera)) : d = null, b.pop(), b.length > 0 ? m = b[b.length - 1] : m = null;
    };
    function sa(T, O, q, $) {
      if (T.visible === !1) return;
      if (T.layers.test(O.layers)) {
        if (T.isGroup)
          q = T.renderOrder;
        else if (T.isLOD)
          T.autoUpdate === !0 && T.update(O);
        else if (T.isLight)
          d.pushLight(T), T.castShadow && d.pushShadow(T);
        else if (T.isSprite) {
          if (!T.frustumCulled || Ge.intersectsSprite(T)) {
            $ && Be.setFromMatrixPosition(T.matrixWorld).applyMatrix4(re);
            const Me = I.update(T), De = T.material;
            De.visible && m.push(T, Me, De, q, Be.z, null);
          }
        } else if ((T.isMesh || T.isLine || T.isPoints) && (!T.frustumCulled || Ge.intersectsObject(T))) {
          const Me = I.update(T), De = T.material;
          if ($ && (T.boundingSphere !== void 0 ? (T.boundingSphere === null && T.computeBoundingSphere(), Be.copy(T.boundingSphere.center)) : (Me.boundingSphere === null && Me.computeBoundingSphere(), Be.copy(Me.boundingSphere.center)), Be.applyMatrix4(T.matrixWorld).applyMatrix4(re)), Array.isArray(De)) {
            const we = Me.groups;
            for (let Oe = 0, He = we.length; Oe < He; Oe++) {
              const Ie = we[Oe], $e = De[Ie.materialIndex];
              $e && $e.visible && m.push(T, Me, $e, q, Be.z, Ie);
            }
          } else De.visible && m.push(T, Me, De, q, Be.z, null);
        }
      }
      const ue = T.children;
      for (let Me = 0, De = ue.length; Me < De; Me++)
        sa(ue[Me], O, q, $);
    }
    function Vc(T, O, q, $) {
      const B = T.opaque, ue = T.transmissive, Me = T.transparent;
      d.setupLightsView(q), Ae === !0 && oe.setGlobalState(M.clippingPlanes, q), $ && z.viewport(D.copy($)), B.length > 0 && Or(B, O, q), ue.length > 0 && Or(ue, O, q), Me.length > 0 && Or(Me, O, q), z.buffers.depth.setTest(!0), z.buffers.depth.setMask(!0), z.buffers.color.setMask(!0), z.setPolygonOffset(!1);
    }
    function kc(T, O, q, $) {
      if ((q.isScene === !0 ? q.overrideMaterial : null) !== null)
        return;
      d.state.transmissionRenderTarget[$.id] === void 0 && (d.state.transmissionRenderTarget[$.id] = new ji(1, 1, {
        generateMipmaps: !0,
        type: K.has("EXT_color_buffer_half_float") || K.has("EXT_color_buffer_float") ? Ir : Bn,
        minFilter: ki,
        samples: 4,
        stencilBuffer: r,
        resolveDepthBuffer: !1,
        resolveStencilBuffer: !1,
        colorSpace: et.workingColorSpace
      }));
      const ue = d.state.transmissionRenderTarget[$.id], Me = $.viewport || D;
      ue.setSize(Me.z * M.transmissionResolutionScale, Me.w * M.transmissionResolutionScale);
      const De = M.getRenderTarget(), we = M.getActiveCubeFace(), Oe = M.getActiveMipmapLevel();
      M.setRenderTarget(ue), M.getClearColor(Z), ne = M.getClearAlpha(), ne < 1 && M.setClearColor(16777215, 0.5), M.clear(), Ze && Ee.render(q);
      const He = M.toneMapping;
      M.toneMapping = Mi;
      const Ie = $.viewport;
      if ($.viewport !== void 0 && ($.viewport = void 0), d.setupLightsView($), Ae === !0 && oe.setGlobalState(M.clippingPlanes, $), Or(T, q, $), ee.updateMultisampleRenderTarget(ue), ee.updateRenderTargetMipmap(ue), K.has("WEBGL_multisampled_render_to_texture") === !1) {
        let $e = !1;
        for (let rt = 0, Mt = O.length; rt < Mt; rt++) {
          const dt = O[rt], ct = dt.object, Ue = dt.geometry, _t = dt.material, Qe = dt.group;
          if (_t.side === Qn && ct.layers.test($.layers)) {
            const Jt = _t.side;
            _t.side = Wt, _t.needsUpdate = !0, Gc(ct, q, $, Ue, _t, Qe), _t.side = Jt, _t.needsUpdate = !0, $e = !0;
          }
        }
        $e === !0 && (ee.updateMultisampleRenderTarget(ue), ee.updateRenderTargetMipmap(ue));
      }
      M.setRenderTarget(De, we, Oe), M.setClearColor(Z, ne), Ie !== void 0 && ($.viewport = Ie), M.toneMapping = He;
    }
    function Or(T, O, q) {
      const $ = O.isScene === !0 ? O.overrideMaterial : null;
      for (let B = 0, ue = T.length; B < ue; B++) {
        const Me = T[B], De = Me.object, we = Me.geometry, Oe = Me.group;
        let He = Me.material;
        He.allowOverride === !0 && $ !== null && (He = $), De.layers.test(q.layers) && Gc(De, O, q, we, He, Oe);
      }
    }
    function Gc(T, O, q, $, B, ue) {
      T.onBeforeRender(M, O, q, $, B, ue), T.modelViewMatrix.multiplyMatrices(q.matrixWorldInverse, T.matrixWorld), T.normalMatrix.getNormalMatrix(T.modelViewMatrix), B.onBeforeRender(M, O, q, $, T, ue), B.transparent === !0 && B.side === Qn && B.forceSinglePass === !1 ? (B.side = Wt, B.needsUpdate = !0, M.renderBufferDirect(q, O, $, B, T, ue), B.side = yi, B.needsUpdate = !0, M.renderBufferDirect(q, O, $, B, T, ue), B.side = Qn) : M.renderBufferDirect(q, O, $, B, T, ue), T.onAfterRender(M, O, q, $, B, ue);
    }
    function Br(T, O, q) {
      O.isScene !== !0 && (O = Pe);
      const $ = j.get(T), B = d.state.lights, ue = d.state.shadowsArray, Me = B.state.version, De = k.getParameters(T, B.state, ue, O, q), we = k.getProgramCacheKey(De);
      let Oe = $.programs;
      $.environment = T.isMeshStandardMaterial ? O.environment : null, $.fog = O.fog, $.envMap = (T.isMeshStandardMaterial ? xe : te).get(T.envMap || $.environment), $.envMapRotation = $.environment !== null && T.envMap === null ? O.environmentRotation : T.envMapRotation, Oe === void 0 && (T.addEventListener("dispose", se), Oe = /* @__PURE__ */ new Map(), $.programs = Oe);
      let He = Oe.get(we);
      if (He !== void 0) {
        if ($.currentProgram === He && $.lightsStateVersion === Me)
          return Xc(T, De), He;
      } else
        De.uniforms = k.getUniforms(T), T.onBeforeCompile(De, M), He = k.acquireProgram(De, we), Oe.set(we, He), $.uniforms = De.uniforms;
      const Ie = $.uniforms;
      return (!T.isShaderMaterial && !T.isRawShaderMaterial || T.clipping === !0) && (Ie.clippingPlanes = oe.uniform), Xc(T, De), $.needsLights = Gd(T), $.lightsStateVersion = Me, $.needsLights && (Ie.ambientLightColor.value = B.state.ambient, Ie.lightProbe.value = B.state.probe, Ie.directionalLights.value = B.state.directional, Ie.directionalLightShadows.value = B.state.directionalShadow, Ie.spotLights.value = B.state.spot, Ie.spotLightShadows.value = B.state.spotShadow, Ie.rectAreaLights.value = B.state.rectArea, Ie.ltc_1.value = B.state.rectAreaLTC1, Ie.ltc_2.value = B.state.rectAreaLTC2, Ie.pointLights.value = B.state.point, Ie.pointLightShadows.value = B.state.pointShadow, Ie.hemisphereLights.value = B.state.hemi, Ie.directionalShadowMap.value = B.state.directionalShadowMap, Ie.directionalShadowMatrix.value = B.state.directionalShadowMatrix, Ie.spotShadowMap.value = B.state.spotShadowMap, Ie.spotLightMatrix.value = B.state.spotLightMatrix, Ie.spotLightMap.value = B.state.spotLightMap, Ie.pointShadowMap.value = B.state.pointShadowMap, Ie.pointShadowMatrix.value = B.state.pointShadowMatrix), $.currentProgram = He, $.uniformsList = null, He;
    }
    function Wc(T) {
      if (T.uniformsList === null) {
        const O = T.currentProgram.getUniforms();
        T.uniformsList = Ao.seqWithValue(O.seq, T.uniforms);
      }
      return T.uniformsList;
    }
    function Xc(T, O) {
      const q = j.get(T);
      q.outputColorSpace = O.outputColorSpace, q.batching = O.batching, q.batchingColor = O.batchingColor, q.instancing = O.instancing, q.instancingColor = O.instancingColor, q.instancingMorph = O.instancingMorph, q.skinning = O.skinning, q.morphTargets = O.morphTargets, q.morphNormals = O.morphNormals, q.morphColors = O.morphColors, q.morphTargetsCount = O.morphTargetsCount, q.numClippingPlanes = O.numClippingPlanes, q.numIntersection = O.numClipIntersection, q.vertexAlphas = O.vertexAlphas, q.vertexTangents = O.vertexTangents, q.toneMapping = O.toneMapping;
    }
    function Vd(T, O, q, $, B) {
      O.isScene !== !0 && (O = Pe), ee.resetTextureUnits();
      const ue = O.fog, Me = $.isMeshStandardMaterial ? O.environment : null, De = U === null ? M.outputColorSpace : U.isXRRenderTarget === !0 ? U.texture.colorSpace : Bs, we = ($.isMeshStandardMaterial ? xe : te).get($.envMap || Me), Oe = $.vertexColors === !0 && !!q.attributes.color && q.attributes.color.itemSize === 4, He = !!q.attributes.tangent && (!!$.normalMap || $.anisotropy > 0), Ie = !!q.morphAttributes.position, $e = !!q.morphAttributes.normal, rt = !!q.morphAttributes.color;
      let Mt = Mi;
      $.toneMapped && (U === null || U.isXRRenderTarget === !0) && (Mt = M.toneMapping);
      const dt = q.morphAttributes.position || q.morphAttributes.normal || q.morphAttributes.color, ct = dt !== void 0 ? dt.length : 0, Ue = j.get($), _t = d.state.lights;
      if (Ae === !0 && (X === !0 || T !== y)) {
        const Ft = T === y && $.id === S;
        oe.setState($, T, Ft);
      }
      let Qe = !1;
      $.version === Ue.__version ? (Ue.needsLights && Ue.lightsStateVersion !== _t.state.version || Ue.outputColorSpace !== De || B.isBatchedMesh && Ue.batching === !1 || !B.isBatchedMesh && Ue.batching === !0 || B.isBatchedMesh && Ue.batchingColor === !0 && B.colorTexture === null || B.isBatchedMesh && Ue.batchingColor === !1 && B.colorTexture !== null || B.isInstancedMesh && Ue.instancing === !1 || !B.isInstancedMesh && Ue.instancing === !0 || B.isSkinnedMesh && Ue.skinning === !1 || !B.isSkinnedMesh && Ue.skinning === !0 || B.isInstancedMesh && Ue.instancingColor === !0 && B.instanceColor === null || B.isInstancedMesh && Ue.instancingColor === !1 && B.instanceColor !== null || B.isInstancedMesh && Ue.instancingMorph === !0 && B.morphTexture === null || B.isInstancedMesh && Ue.instancingMorph === !1 && B.morphTexture !== null || Ue.envMap !== we || $.fog === !0 && Ue.fog !== ue || Ue.numClippingPlanes !== void 0 && (Ue.numClippingPlanes !== oe.numPlanes || Ue.numIntersection !== oe.numIntersection) || Ue.vertexAlphas !== Oe || Ue.vertexTangents !== He || Ue.morphTargets !== Ie || Ue.morphNormals !== $e || Ue.morphColors !== rt || Ue.toneMapping !== Mt || Ue.morphTargetsCount !== ct) && (Qe = !0) : (Qe = !0, Ue.__version = $.version);
      let Jt = Ue.currentProgram;
      Qe === !0 && (Jt = Br($, O, B));
      let es = !1, Qt = !1, Gs = !1;
      const gt = Jt.getUniforms(), an = Ue.uniforms;
      if (z.useProgram(Jt.program) && (es = !0, Qt = !0, Gs = !0), $.id !== S && (S = $.id, Qt = !0), es || y !== T) {
        z.buffers.depth.getReversed() && T.reversedDepth !== !0 && (T._reversedDepth = !0, T.updateProjectionMatrix()), gt.setValue(g, "projectionMatrix", T.projectionMatrix), gt.setValue(g, "viewMatrix", T.matrixWorldInverse);
        const Xt = gt.map.cameraPosition;
        Xt !== void 0 && Xt.setValue(g, be.setFromMatrixPosition(T.matrixWorld)), Y.logarithmicDepthBuffer && gt.setValue(
          g,
          "logDepthBufFC",
          2 / (Math.log(T.far + 1) / Math.LN2)
        ), ($.isMeshPhongMaterial || $.isMeshToonMaterial || $.isMeshLambertMaterial || $.isMeshBasicMaterial || $.isMeshStandardMaterial || $.isShaderMaterial) && gt.setValue(g, "isOrthographic", T.isOrthographicCamera === !0), y !== T && (y = T, Qt = !0, Gs = !0);
      }
      if (B.isSkinnedMesh) {
        gt.setOptional(g, B, "bindMatrix"), gt.setOptional(g, B, "bindMatrixInverse");
        const Ft = B.skeleton;
        Ft && (Ft.boneTexture === null && Ft.computeBoneTexture(), gt.setValue(g, "boneTexture", Ft.boneTexture, ee));
      }
      B.isBatchedMesh && (gt.setOptional(g, B, "batchingTexture"), gt.setValue(g, "batchingTexture", B._matricesTexture, ee), gt.setOptional(g, B, "batchingIdTexture"), gt.setValue(g, "batchingIdTexture", B._indirectTexture, ee), gt.setOptional(g, B, "batchingColorTexture"), B._colorsTexture !== null && gt.setValue(g, "batchingColorTexture", B._colorsTexture, ee));
      const ln = q.morphAttributes;
      if ((ln.position !== void 0 || ln.normal !== void 0 || ln.color !== void 0) && le.update(B, q, Jt), (Qt || Ue.receiveShadow !== B.receiveShadow) && (Ue.receiveShadow = B.receiveShadow, gt.setValue(g, "receiveShadow", B.receiveShadow)), $.isMeshGouraudMaterial && $.envMap !== null && (an.envMap.value = we, an.flipEnvMap.value = we.isCubeTexture && we.isRenderTargetTexture === !1 ? -1 : 1), $.isMeshStandardMaterial && $.envMap === null && O.environment !== null && (an.envMapIntensity.value = O.environmentIntensity), Qt && (gt.setValue(g, "toneMappingExposure", M.toneMappingExposure), Ue.needsLights && kd(an, Gs), ue && $.fog === !0 && Q.refreshFogUniforms(an, ue), Q.refreshMaterialUniforms(an, $, H, ie, d.state.transmissionRenderTarget[T.id]), Ao.upload(g, Wc(Ue), an, ee)), $.isShaderMaterial && $.uniformsNeedUpdate === !0 && (Ao.upload(g, Wc(Ue), an, ee), $.uniformsNeedUpdate = !1), $.isSpriteMaterial && gt.setValue(g, "center", B.center), gt.setValue(g, "modelViewMatrix", B.modelViewMatrix), gt.setValue(g, "normalMatrix", B.normalMatrix), gt.setValue(g, "modelMatrix", B.matrixWorld), $.isShaderMaterial || $.isRawShaderMaterial) {
        const Ft = $.uniformsGroups;
        for (let Xt = 0, ra = Ft.length; Xt < ra; Xt++) {
          const bi = Ft[Xt];
          ke.update(bi, Jt), ke.bind(bi, Jt);
        }
      }
      return Jt;
    }
    function kd(T, O) {
      T.ambientLightColor.needsUpdate = O, T.lightProbe.needsUpdate = O, T.directionalLights.needsUpdate = O, T.directionalLightShadows.needsUpdate = O, T.pointLights.needsUpdate = O, T.pointLightShadows.needsUpdate = O, T.spotLights.needsUpdate = O, T.spotLightShadows.needsUpdate = O, T.rectAreaLights.needsUpdate = O, T.hemisphereLights.needsUpdate = O;
    }
    function Gd(T) {
      return T.isMeshLambertMaterial || T.isMeshToonMaterial || T.isMeshPhongMaterial || T.isMeshStandardMaterial || T.isShadowMaterial || T.isShaderMaterial && T.lights === !0;
    }
    this.getActiveCubeFace = function() {
      return w;
    }, this.getActiveMipmapLevel = function() {
      return P;
    }, this.getRenderTarget = function() {
      return U;
    }, this.setRenderTargetTextures = function(T, O, q) {
      const $ = j.get(T);
      $.__autoAllocateDepthBuffer = T.resolveDepthBuffer === !1, $.__autoAllocateDepthBuffer === !1 && ($.__useRenderToTexture = !1), j.get(T.texture).__webglTexture = O, j.get(T.depthTexture).__webglTexture = $.__autoAllocateDepthBuffer ? void 0 : q, $.__hasExternalTextures = !0;
    }, this.setRenderTargetFramebuffer = function(T, O) {
      const q = j.get(T);
      q.__webglFramebuffer = O, q.__useDefaultFramebuffer = O === void 0;
    };
    const Wd = g.createFramebuffer();
    this.setRenderTarget = function(T, O = 0, q = 0) {
      U = T, w = O, P = q;
      let $ = !0, B = null, ue = !1, Me = !1;
      if (T) {
        const we = j.get(T);
        if (we.__useDefaultFramebuffer !== void 0)
          z.bindFramebuffer(g.FRAMEBUFFER, null), $ = !1;
        else if (we.__webglFramebuffer === void 0)
          ee.setupRenderTarget(T);
        else if (we.__hasExternalTextures)
          ee.rebindTextures(T, j.get(T.texture).__webglTexture, j.get(T.depthTexture).__webglTexture);
        else if (T.depthBuffer) {
          const Ie = T.depthTexture;
          if (we.__boundDepthTexture !== Ie) {
            if (Ie !== null && j.has(Ie) && (T.width !== Ie.image.width || T.height !== Ie.image.height))
              throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");
            ee.setupDepthRenderbuffer(T);
          }
        }
        const Oe = T.texture;
        (Oe.isData3DTexture || Oe.isDataArrayTexture || Oe.isCompressedArrayTexture) && (Me = !0);
        const He = j.get(T).__webglFramebuffer;
        T.isWebGLCubeRenderTarget ? (Array.isArray(He[O]) ? B = He[O][q] : B = He[O], ue = !0) : T.samples > 0 && ee.useMultisampledRTT(T) === !1 ? B = j.get(T).__webglMultisampledFramebuffer : Array.isArray(He) ? B = He[q] : B = He, D.copy(T.viewport), L.copy(T.scissor), V = T.scissorTest;
      } else
        D.copy(ye).multiplyScalar(H).floor(), L.copy(Fe).multiplyScalar(H).floor(), V = Je;
      if (q !== 0 && (B = Wd), z.bindFramebuffer(g.FRAMEBUFFER, B) && $ && z.drawBuffers(T, B), z.viewport(D), z.scissor(L), z.setScissorTest(V), ue) {
        const we = j.get(T.texture);
        g.framebufferTexture2D(g.FRAMEBUFFER, g.COLOR_ATTACHMENT0, g.TEXTURE_CUBE_MAP_POSITIVE_X + O, we.__webglTexture, q);
      } else if (Me) {
        const we = O;
        for (let Oe = 0; Oe < T.textures.length; Oe++) {
          const He = j.get(T.textures[Oe]);
          g.framebufferTextureLayer(g.FRAMEBUFFER, g.COLOR_ATTACHMENT0 + Oe, He.__webglTexture, q, we);
        }
      } else if (T !== null && q !== 0) {
        const we = j.get(T.texture);
        g.framebufferTexture2D(g.FRAMEBUFFER, g.COLOR_ATTACHMENT0, g.TEXTURE_2D, we.__webglTexture, q);
      }
      S = -1;
    }, this.readRenderTargetPixels = function(T, O, q, $, B, ue, Me, De = 0) {
      if (!(T && T.isWebGLRenderTarget)) {
        console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");
        return;
      }
      let we = j.get(T).__webglFramebuffer;
      if (T.isWebGLCubeRenderTarget && Me !== void 0 && (we = we[Me]), we) {
        z.bindFramebuffer(g.FRAMEBUFFER, we);
        try {
          const Oe = T.textures[De], He = Oe.format, Ie = Oe.type;
          if (!Y.textureFormatReadable(He)) {
            console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");
            return;
          }
          if (!Y.textureTypeReadable(Ie)) {
            console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");
            return;
          }
          O >= 0 && O <= T.width - $ && q >= 0 && q <= T.height - B && (T.textures.length > 1 && g.readBuffer(g.COLOR_ATTACHMENT0 + De), g.readPixels(O, q, $, B, Te.convert(He), Te.convert(Ie), ue));
        } finally {
          const Oe = U !== null ? j.get(U).__webglFramebuffer : null;
          z.bindFramebuffer(g.FRAMEBUFFER, Oe);
        }
      }
    }, this.readRenderTargetPixelsAsync = async function(T, O, q, $, B, ue, Me, De = 0) {
      if (!(T && T.isWebGLRenderTarget))
        throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");
      let we = j.get(T).__webglFramebuffer;
      if (T.isWebGLCubeRenderTarget && Me !== void 0 && (we = we[Me]), we)
        if (O >= 0 && O <= T.width - $ && q >= 0 && q <= T.height - B) {
          z.bindFramebuffer(g.FRAMEBUFFER, we);
          const Oe = T.textures[De], He = Oe.format, Ie = Oe.type;
          if (!Y.textureFormatReadable(He))
            throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");
          if (!Y.textureTypeReadable(Ie))
            throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");
          const $e = g.createBuffer();
          g.bindBuffer(g.PIXEL_PACK_BUFFER, $e), g.bufferData(g.PIXEL_PACK_BUFFER, ue.byteLength, g.STREAM_READ), T.textures.length > 1 && g.readBuffer(g.COLOR_ATTACHMENT0 + De), g.readPixels(O, q, $, B, Te.convert(He), Te.convert(Ie), 0);
          const rt = U !== null ? j.get(U).__webglFramebuffer : null;
          z.bindFramebuffer(g.FRAMEBUFFER, rt);
          const Mt = g.fenceSync(g.SYNC_GPU_COMMANDS_COMPLETE, 0);
          return g.flush(), await Ug(g, Mt, 4), g.bindBuffer(g.PIXEL_PACK_BUFFER, $e), g.getBufferSubData(g.PIXEL_PACK_BUFFER, 0, ue), g.deleteBuffer($e), g.deleteSync(Mt), ue;
        } else
          throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.");
    }, this.copyFramebufferToTexture = function(T, O = null, q = 0) {
      const $ = Math.pow(2, -q), B = Math.floor(T.image.width * $), ue = Math.floor(T.image.height * $), Me = O !== null ? O.x : 0, De = O !== null ? O.y : 0;
      ee.setTexture2D(T, 0), g.copyTexSubImage2D(g.TEXTURE_2D, q, 0, 0, Me, De, B, ue), z.unbindTexture();
    };
    const Xd = g.createFramebuffer(), Yd = g.createFramebuffer();
    this.copyTextureToTexture = function(T, O, q = null, $ = null, B = 0, ue = null) {
      ue === null && (B !== 0 ? (Cr("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."), ue = B, B = 0) : ue = 0);
      let Me, De, we, Oe, He, Ie, $e, rt, Mt;
      const dt = T.isCompressedTexture ? T.mipmaps[ue] : T.image;
      if (q !== null)
        Me = q.max.x - q.min.x, De = q.max.y - q.min.y, we = q.isBox3 ? q.max.z - q.min.z : 1, Oe = q.min.x, He = q.min.y, Ie = q.isBox3 ? q.min.z : 0;
      else {
        const ln = Math.pow(2, -B);
        Me = Math.floor(dt.width * ln), De = Math.floor(dt.height * ln), T.isDataArrayTexture ? we = dt.depth : T.isData3DTexture ? we = Math.floor(dt.depth * ln) : we = 1, Oe = 0, He = 0, Ie = 0;
      }
      $ !== null ? ($e = $.x, rt = $.y, Mt = $.z) : ($e = 0, rt = 0, Mt = 0);
      const ct = Te.convert(O.format), Ue = Te.convert(O.type);
      let _t;
      O.isData3DTexture ? (ee.setTexture3D(O, 0), _t = g.TEXTURE_3D) : O.isDataArrayTexture || O.isCompressedArrayTexture ? (ee.setTexture2DArray(O, 0), _t = g.TEXTURE_2D_ARRAY) : (ee.setTexture2D(O, 0), _t = g.TEXTURE_2D), g.pixelStorei(g.UNPACK_FLIP_Y_WEBGL, O.flipY), g.pixelStorei(g.UNPACK_PREMULTIPLY_ALPHA_WEBGL, O.premultiplyAlpha), g.pixelStorei(g.UNPACK_ALIGNMENT, O.unpackAlignment);
      const Qe = g.getParameter(g.UNPACK_ROW_LENGTH), Jt = g.getParameter(g.UNPACK_IMAGE_HEIGHT), es = g.getParameter(g.UNPACK_SKIP_PIXELS), Qt = g.getParameter(g.UNPACK_SKIP_ROWS), Gs = g.getParameter(g.UNPACK_SKIP_IMAGES);
      g.pixelStorei(g.UNPACK_ROW_LENGTH, dt.width), g.pixelStorei(g.UNPACK_IMAGE_HEIGHT, dt.height), g.pixelStorei(g.UNPACK_SKIP_PIXELS, Oe), g.pixelStorei(g.UNPACK_SKIP_ROWS, He), g.pixelStorei(g.UNPACK_SKIP_IMAGES, Ie);
      const gt = T.isDataArrayTexture || T.isData3DTexture, an = O.isDataArrayTexture || O.isData3DTexture;
      if (T.isDepthTexture) {
        const ln = j.get(T), Ft = j.get(O), Xt = j.get(ln.__renderTarget), ra = j.get(Ft.__renderTarget);
        z.bindFramebuffer(g.READ_FRAMEBUFFER, Xt.__webglFramebuffer), z.bindFramebuffer(g.DRAW_FRAMEBUFFER, ra.__webglFramebuffer);
        for (let bi = 0; bi < we; bi++)
          gt && (g.framebufferTextureLayer(g.READ_FRAMEBUFFER, g.COLOR_ATTACHMENT0, j.get(T).__webglTexture, B, Ie + bi), g.framebufferTextureLayer(g.DRAW_FRAMEBUFFER, g.COLOR_ATTACHMENT0, j.get(O).__webglTexture, ue, Mt + bi)), g.blitFramebuffer(Oe, He, Me, De, $e, rt, Me, De, g.DEPTH_BUFFER_BIT, g.NEAREST);
        z.bindFramebuffer(g.READ_FRAMEBUFFER, null), z.bindFramebuffer(g.DRAW_FRAMEBUFFER, null);
      } else if (B !== 0 || T.isRenderTargetTexture || j.has(T)) {
        const ln = j.get(T), Ft = j.get(O);
        z.bindFramebuffer(g.READ_FRAMEBUFFER, Xd), z.bindFramebuffer(g.DRAW_FRAMEBUFFER, Yd);
        for (let Xt = 0; Xt < we; Xt++)
          gt ? g.framebufferTextureLayer(g.READ_FRAMEBUFFER, g.COLOR_ATTACHMENT0, ln.__webglTexture, B, Ie + Xt) : g.framebufferTexture2D(g.READ_FRAMEBUFFER, g.COLOR_ATTACHMENT0, g.TEXTURE_2D, ln.__webglTexture, B), an ? g.framebufferTextureLayer(g.DRAW_FRAMEBUFFER, g.COLOR_ATTACHMENT0, Ft.__webglTexture, ue, Mt + Xt) : g.framebufferTexture2D(g.DRAW_FRAMEBUFFER, g.COLOR_ATTACHMENT0, g.TEXTURE_2D, Ft.__webglTexture, ue), B !== 0 ? g.blitFramebuffer(Oe, He, Me, De, $e, rt, Me, De, g.COLOR_BUFFER_BIT, g.NEAREST) : an ? g.copyTexSubImage3D(_t, ue, $e, rt, Mt + Xt, Oe, He, Me, De) : g.copyTexSubImage2D(_t, ue, $e, rt, Oe, He, Me, De);
        z.bindFramebuffer(g.READ_FRAMEBUFFER, null), z.bindFramebuffer(g.DRAW_FRAMEBUFFER, null);
      } else
        an ? T.isDataTexture || T.isData3DTexture ? g.texSubImage3D(_t, ue, $e, rt, Mt, Me, De, we, ct, Ue, dt.data) : O.isCompressedArrayTexture ? g.compressedTexSubImage3D(_t, ue, $e, rt, Mt, Me, De, we, ct, dt.data) : g.texSubImage3D(_t, ue, $e, rt, Mt, Me, De, we, ct, Ue, dt) : T.isDataTexture ? g.texSubImage2D(g.TEXTURE_2D, ue, $e, rt, Me, De, ct, Ue, dt.data) : T.isCompressedTexture ? g.compressedTexSubImage2D(g.TEXTURE_2D, ue, $e, rt, dt.width, dt.height, ct, dt.data) : g.texSubImage2D(g.TEXTURE_2D, ue, $e, rt, Me, De, ct, Ue, dt);
      g.pixelStorei(g.UNPACK_ROW_LENGTH, Qe), g.pixelStorei(g.UNPACK_IMAGE_HEIGHT, Jt), g.pixelStorei(g.UNPACK_SKIP_PIXELS, es), g.pixelStorei(g.UNPACK_SKIP_ROWS, Qt), g.pixelStorei(g.UNPACK_SKIP_IMAGES, Gs), ue === 0 && O.generateMipmaps && g.generateMipmap(_t), z.unbindTexture();
    }, this.initRenderTarget = function(T) {
      j.get(T).__webglFramebuffer === void 0 && ee.setupRenderTarget(T);
    }, this.initTexture = function(T) {
      T.isCubeTexture ? ee.setTextureCube(T, 0) : T.isData3DTexture ? ee.setTexture3D(T, 0) : T.isDataArrayTexture || T.isCompressedArrayTexture ? ee.setTexture2DArray(T, 0) : ee.setTexture2D(T, 0), z.unbindTexture();
    }, this.resetState = function() {
      w = 0, P = 0, U = null, z.reset(), me.reset();
    }, typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe", { detail: this }));
  }
  /**
   * Defines the coordinate system of the renderer.
   *
   * In `WebGLRenderer`, the value is always `WebGLCoordinateSystem`.
   *
   * @type {WebGLCoordinateSystem|WebGPUCoordinateSystem}
   * @default WebGLCoordinateSystem
   * @readonly
   */
  get coordinateSystem() {
    return Nn;
  }
  /**
   * Defines the output color space of the renderer.
   *
   * @type {SRGBColorSpace|LinearSRGBColorSpace}
   * @default SRGBColorSpace
   */
  get outputColorSpace() {
    return this._outputColorSpace;
  }
  set outputColorSpace(e) {
    this._outputColorSpace = e;
    const t = this.getContext();
    t.drawingBufferColorSpace = et._getDrawingBufferColorSpace(e), t.unpackColorSpace = et._getUnpackColorSpace();
  }
}
const Ih = { type: "change" }, Uc = { type: "start" }, Ud = { type: "end" }, go = new na(), Uh = new mi(), ry = Math.cos(70 * Lg.DEG2RAD), Et = new N(), qt = 2 * Math.PI, at = {
  NONE: -1,
  ROTATE: 0,
  DOLLY: 1,
  PAN: 2,
  TOUCH_ROTATE: 3,
  TOUCH_PAN: 4,
  TOUCH_DOLLY_PAN: 5,
  TOUCH_DOLLY_ROTATE: 6
}, Qa = 1e-6;
class oy extends S0 {
  /**
   * Constructs a new controls instance.
   *
   * @param {Object3D} object - The object that is managed by the controls.
   * @param {?HTMLDOMElement} domElement - The HTML element used for event listeners.
   */
  constructor(e, t = null) {
    super(e, t), this.state = at.NONE, this.target = new N(), this.cursor = new N(), this.minDistance = 0, this.maxDistance = 1 / 0, this.minZoom = 0, this.maxZoom = 1 / 0, this.minTargetRadius = 0, this.maxTargetRadius = 1 / 0, this.minPolarAngle = 0, this.maxPolarAngle = Math.PI, this.minAzimuthAngle = -1 / 0, this.maxAzimuthAngle = 1 / 0, this.enableDamping = !1, this.dampingFactor = 0.05, this.enableZoom = !0, this.zoomSpeed = 1, this.enableRotate = !0, this.rotateSpeed = 1, this.keyRotateSpeed = 1, this.enablePan = !0, this.panSpeed = 1, this.screenSpacePanning = !0, this.keyPanSpeed = 7, this.zoomToCursor = !1, this.autoRotate = !1, this.autoRotateSpeed = 2, this.keys = { LEFT: "ArrowLeft", UP: "ArrowUp", RIGHT: "ArrowRight", BOTTOM: "ArrowDown" }, this.mouseButtons = { LEFT: Ds.ROTATE, MIDDLE: Ds.DOLLY, RIGHT: Ds.PAN }, this.touches = { ONE: ys.ROTATE, TWO: ys.DOLLY_PAN }, this.target0 = this.target.clone(), this.position0 = this.object.position.clone(), this.zoom0 = this.object.zoom, this._domElementKeyEvents = null, this._lastPosition = new N(), this._lastQuaternion = new qi(), this._lastTargetPosition = new N(), this._quat = new qi().setFromUnitVectors(e.up, new N(0, 1, 0)), this._quatInverse = this._quat.clone().invert(), this._spherical = new oh(), this._sphericalDelta = new oh(), this._scale = 1, this._panOffset = new N(), this._rotateStart = new Ve(), this._rotateEnd = new Ve(), this._rotateDelta = new Ve(), this._panStart = new Ve(), this._panEnd = new Ve(), this._panDelta = new Ve(), this._dollyStart = new Ve(), this._dollyEnd = new Ve(), this._dollyDelta = new Ve(), this._dollyDirection = new N(), this._mouse = new Ve(), this._performCursorZoom = !1, this._pointers = [], this._pointerPositions = {}, this._controlActive = !1, this._onPointerMove = ly.bind(this), this._onPointerDown = ay.bind(this), this._onPointerUp = cy.bind(this), this._onContextMenu = _y.bind(this), this._onMouseWheel = fy.bind(this), this._onKeyDown = dy.bind(this), this._onTouchStart = py.bind(this), this._onTouchMove = my.bind(this), this._onMouseDown = uy.bind(this), this._onMouseMove = hy.bind(this), this._interceptControlDown = gy.bind(this), this._interceptControlUp = vy.bind(this), this.domElement !== null && this.connect(this.domElement), this.update();
  }
  connect(e) {
    super.connect(e), this.domElement.addEventListener("pointerdown", this._onPointerDown), this.domElement.addEventListener("pointercancel", this._onPointerUp), this.domElement.addEventListener("contextmenu", this._onContextMenu), this.domElement.addEventListener("wheel", this._onMouseWheel, { passive: !1 }), this.domElement.getRootNode().addEventListener("keydown", this._interceptControlDown, { passive: !0, capture: !0 }), this.domElement.style.touchAction = "none";
  }
  disconnect() {
    this.domElement.removeEventListener("pointerdown", this._onPointerDown), this.domElement.removeEventListener("pointermove", this._onPointerMove), this.domElement.removeEventListener("pointerup", this._onPointerUp), this.domElement.removeEventListener("pointercancel", this._onPointerUp), this.domElement.removeEventListener("wheel", this._onMouseWheel), this.domElement.removeEventListener("contextmenu", this._onContextMenu), this.stopListenToKeyEvents(), this.domElement.getRootNode().removeEventListener("keydown", this._interceptControlDown, { capture: !0 }), this.domElement.style.touchAction = "auto";
  }
  dispose() {
    this.disconnect();
  }
  /**
   * Get the current vertical rotation, in radians.
   *
   * @return {number} The current vertical rotation, in radians.
   */
  getPolarAngle() {
    return this._spherical.phi;
  }
  /**
   * Get the current horizontal rotation, in radians.
   *
   * @return {number} The current horizontal rotation, in radians.
   */
  getAzimuthalAngle() {
    return this._spherical.theta;
  }
  /**
   * Returns the distance from the camera to the target.
   *
   * @return {number} The distance from the camera to the target.
   */
  getDistance() {
    return this.object.position.distanceTo(this.target);
  }
  /**
   * Adds key event listeners to the given DOM element.
   * `window` is a recommended argument for using this method.
   *
   * @param {HTMLDOMElement} domElement - The DOM element
   */
  listenToKeyEvents(e) {
    e.addEventListener("keydown", this._onKeyDown), this._domElementKeyEvents = e;
  }
  /**
   * Removes the key event listener previously defined with `listenToKeyEvents()`.
   */
  stopListenToKeyEvents() {
    this._domElementKeyEvents !== null && (this._domElementKeyEvents.removeEventListener("keydown", this._onKeyDown), this._domElementKeyEvents = null);
  }
  /**
   * Save the current state of the controls. This can later be recovered with `reset()`.
   */
  saveState() {
    this.target0.copy(this.target), this.position0.copy(this.object.position), this.zoom0 = this.object.zoom;
  }
  /**
   * Reset the controls to their state from either the last time the `saveState()`
   * was called, or the initial state.
   */
  reset() {
    this.target.copy(this.target0), this.object.position.copy(this.position0), this.object.zoom = this.zoom0, this.object.updateProjectionMatrix(), this.dispatchEvent(Ih), this.update(), this.state = at.NONE;
  }
  update(e = null) {
    const t = this.object.position;
    Et.copy(t).sub(this.target), Et.applyQuaternion(this._quat), this._spherical.setFromVector3(Et), this.autoRotate && this.state === at.NONE && this._rotateLeft(this._getAutoRotationAngle(e)), this.enableDamping ? (this._spherical.theta += this._sphericalDelta.theta * this.dampingFactor, this._spherical.phi += this._sphericalDelta.phi * this.dampingFactor) : (this._spherical.theta += this._sphericalDelta.theta, this._spherical.phi += this._sphericalDelta.phi);
    let i = this.minAzimuthAngle, s = this.maxAzimuthAngle;
    isFinite(i) && isFinite(s) && (i < -Math.PI ? i += qt : i > Math.PI && (i -= qt), s < -Math.PI ? s += qt : s > Math.PI && (s -= qt), i <= s ? this._spherical.theta = Math.max(i, Math.min(s, this._spherical.theta)) : this._spherical.theta = this._spherical.theta > (i + s) / 2 ? Math.max(i, this._spherical.theta) : Math.min(s, this._spherical.theta)), this._spherical.phi = Math.max(this.minPolarAngle, Math.min(this.maxPolarAngle, this._spherical.phi)), this._spherical.makeSafe(), this.enableDamping === !0 ? this.target.addScaledVector(this._panOffset, this.dampingFactor) : this.target.add(this._panOffset), this.target.sub(this.cursor), this.target.clampLength(this.minTargetRadius, this.maxTargetRadius), this.target.add(this.cursor);
    let r = !1;
    if (this.zoomToCursor && this._performCursorZoom || this.object.isOrthographicCamera)
      this._spherical.radius = this._clampDistance(this._spherical.radius);
    else {
      const o = this._spherical.radius;
      this._spherical.radius = this._clampDistance(this._spherical.radius * this._scale), r = o != this._spherical.radius;
    }
    if (Et.setFromSpherical(this._spherical), Et.applyQuaternion(this._quatInverse), t.copy(this.target).add(Et), this.object.lookAt(this.target), this.enableDamping === !0 ? (this._sphericalDelta.theta *= 1 - this.dampingFactor, this._sphericalDelta.phi *= 1 - this.dampingFactor, this._panOffset.multiplyScalar(1 - this.dampingFactor)) : (this._sphericalDelta.set(0, 0, 0), this._panOffset.set(0, 0, 0)), this.zoomToCursor && this._performCursorZoom) {
      let o = null;
      if (this.object.isPerspectiveCamera) {
        const a = Et.length();
        o = this._clampDistance(a * this._scale);
        const l = a - o;
        this.object.position.addScaledVector(this._dollyDirection, l), this.object.updateMatrixWorld(), r = !!l;
      } else if (this.object.isOrthographicCamera) {
        const a = new N(this._mouse.x, this._mouse.y, 0);
        a.unproject(this.object);
        const l = this.object.zoom;
        this.object.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.object.zoom / this._scale)), this.object.updateProjectionMatrix(), r = l !== this.object.zoom;
        const c = new N(this._mouse.x, this._mouse.y, 0);
        c.unproject(this.object), this.object.position.sub(c).add(a), this.object.updateMatrixWorld(), o = Et.length();
      } else
        console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."), this.zoomToCursor = !1;
      o !== null && (this.screenSpacePanning ? this.target.set(0, 0, -1).transformDirection(this.object.matrix).multiplyScalar(o).add(this.object.position) : (go.origin.copy(this.object.position), go.direction.set(0, 0, -1).transformDirection(this.object.matrix), Math.abs(this.object.up.dot(go.direction)) < ry ? this.object.lookAt(this.target) : (Uh.setFromNormalAndCoplanarPoint(this.object.up, this.target), go.intersectPlane(Uh, this.target))));
    } else if (this.object.isOrthographicCamera) {
      const o = this.object.zoom;
      this.object.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.object.zoom / this._scale)), o !== this.object.zoom && (this.object.updateProjectionMatrix(), r = !0);
    }
    return this._scale = 1, this._performCursorZoom = !1, r || this._lastPosition.distanceToSquared(this.object.position) > Qa || 8 * (1 - this._lastQuaternion.dot(this.object.quaternion)) > Qa || this._lastTargetPosition.distanceToSquared(this.target) > Qa ? (this.dispatchEvent(Ih), this._lastPosition.copy(this.object.position), this._lastQuaternion.copy(this.object.quaternion), this._lastTargetPosition.copy(this.target), !0) : !1;
  }
  _getAutoRotationAngle(e) {
    return e !== null ? qt / 60 * this.autoRotateSpeed * e : qt / 60 / 60 * this.autoRotateSpeed;
  }
  _getZoomScale(e) {
    const t = Math.abs(e * 0.01);
    return Math.pow(0.95, this.zoomSpeed * t);
  }
  _rotateLeft(e) {
    this._sphericalDelta.theta -= e;
  }
  _rotateUp(e) {
    this._sphericalDelta.phi -= e;
  }
  _panLeft(e, t) {
    Et.setFromMatrixColumn(t, 0), Et.multiplyScalar(-e), this._panOffset.add(Et);
  }
  _panUp(e, t) {
    this.screenSpacePanning === !0 ? Et.setFromMatrixColumn(t, 1) : (Et.setFromMatrixColumn(t, 0), Et.crossVectors(this.object.up, Et)), Et.multiplyScalar(e), this._panOffset.add(Et);
  }
  // deltaX and deltaY are in pixels; right and down are positive
  _pan(e, t) {
    const i = this.domElement;
    if (this.object.isPerspectiveCamera) {
      const s = this.object.position;
      Et.copy(s).sub(this.target);
      let r = Et.length();
      r *= Math.tan(this.object.fov / 2 * Math.PI / 180), this._panLeft(2 * e * r / i.clientHeight, this.object.matrix), this._panUp(2 * t * r / i.clientHeight, this.object.matrix);
    } else this.object.isOrthographicCamera ? (this._panLeft(e * (this.object.right - this.object.left) / this.object.zoom / i.clientWidth, this.object.matrix), this._panUp(t * (this.object.top - this.object.bottom) / this.object.zoom / i.clientHeight, this.object.matrix)) : (console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."), this.enablePan = !1);
  }
  _dollyOut(e) {
    this.object.isPerspectiveCamera || this.object.isOrthographicCamera ? this._scale /= e : (console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."), this.enableZoom = !1);
  }
  _dollyIn(e) {
    this.object.isPerspectiveCamera || this.object.isOrthographicCamera ? this._scale *= e : (console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."), this.enableZoom = !1);
  }
  _updateZoomParameters(e, t) {
    if (!this.zoomToCursor)
      return;
    this._performCursorZoom = !0;
    const i = this.domElement.getBoundingClientRect(), s = e - i.left, r = t - i.top, o = i.width, a = i.height;
    this._mouse.x = s / o * 2 - 1, this._mouse.y = -(r / a) * 2 + 1, this._dollyDirection.set(this._mouse.x, this._mouse.y, 1).unproject(this.object).sub(this.object.position).normalize();
  }
  _clampDistance(e) {
    return Math.max(this.minDistance, Math.min(this.maxDistance, e));
  }
  //
  // event callbacks - update the object state
  //
  _handleMouseDownRotate(e) {
    this._rotateStart.set(e.clientX, e.clientY);
  }
  _handleMouseDownDolly(e) {
    this._updateZoomParameters(e.clientX, e.clientX), this._dollyStart.set(e.clientX, e.clientY);
  }
  _handleMouseDownPan(e) {
    this._panStart.set(e.clientX, e.clientY);
  }
  _handleMouseMoveRotate(e) {
    this._rotateEnd.set(e.clientX, e.clientY), this._rotateDelta.subVectors(this._rotateEnd, this._rotateStart).multiplyScalar(this.rotateSpeed);
    const t = this.domElement;
    this._rotateLeft(qt * this._rotateDelta.x / t.clientHeight), this._rotateUp(qt * this._rotateDelta.y / t.clientHeight), this._rotateStart.copy(this._rotateEnd), this.update();
  }
  _handleMouseMoveDolly(e) {
    this._dollyEnd.set(e.clientX, e.clientY), this._dollyDelta.subVectors(this._dollyEnd, this._dollyStart), this._dollyDelta.y > 0 ? this._dollyOut(this._getZoomScale(this._dollyDelta.y)) : this._dollyDelta.y < 0 && this._dollyIn(this._getZoomScale(this._dollyDelta.y)), this._dollyStart.copy(this._dollyEnd), this.update();
  }
  _handleMouseMovePan(e) {
    this._panEnd.set(e.clientX, e.clientY), this._panDelta.subVectors(this._panEnd, this._panStart).multiplyScalar(this.panSpeed), this._pan(this._panDelta.x, this._panDelta.y), this._panStart.copy(this._panEnd), this.update();
  }
  _handleMouseWheel(e) {
    this._updateZoomParameters(e.clientX, e.clientY), e.deltaY < 0 ? this._dollyIn(this._getZoomScale(e.deltaY)) : e.deltaY > 0 && this._dollyOut(this._getZoomScale(e.deltaY)), this.update();
  }
  _handleKeyDown(e) {
    let t = !1;
    switch (e.code) {
      case this.keys.UP:
        e.ctrlKey || e.metaKey || e.shiftKey ? this.enableRotate && this._rotateUp(qt * this.keyRotateSpeed / this.domElement.clientHeight) : this.enablePan && this._pan(0, this.keyPanSpeed), t = !0;
        break;
      case this.keys.BOTTOM:
        e.ctrlKey || e.metaKey || e.shiftKey ? this.enableRotate && this._rotateUp(-qt * this.keyRotateSpeed / this.domElement.clientHeight) : this.enablePan && this._pan(0, -this.keyPanSpeed), t = !0;
        break;
      case this.keys.LEFT:
        e.ctrlKey || e.metaKey || e.shiftKey ? this.enableRotate && this._rotateLeft(qt * this.keyRotateSpeed / this.domElement.clientHeight) : this.enablePan && this._pan(this.keyPanSpeed, 0), t = !0;
        break;
      case this.keys.RIGHT:
        e.ctrlKey || e.metaKey || e.shiftKey ? this.enableRotate && this._rotateLeft(-qt * this.keyRotateSpeed / this.domElement.clientHeight) : this.enablePan && this._pan(-this.keyPanSpeed, 0), t = !0;
        break;
    }
    t && (e.preventDefault(), this.update());
  }
  _handleTouchStartRotate(e) {
    if (this._pointers.length === 1)
      this._rotateStart.set(e.pageX, e.pageY);
    else {
      const t = this._getSecondPointerPosition(e), i = 0.5 * (e.pageX + t.x), s = 0.5 * (e.pageY + t.y);
      this._rotateStart.set(i, s);
    }
  }
  _handleTouchStartPan(e) {
    if (this._pointers.length === 1)
      this._panStart.set(e.pageX, e.pageY);
    else {
      const t = this._getSecondPointerPosition(e), i = 0.5 * (e.pageX + t.x), s = 0.5 * (e.pageY + t.y);
      this._panStart.set(i, s);
    }
  }
  _handleTouchStartDolly(e) {
    const t = this._getSecondPointerPosition(e), i = e.pageX - t.x, s = e.pageY - t.y, r = Math.sqrt(i * i + s * s);
    this._dollyStart.set(0, r);
  }
  _handleTouchStartDollyPan(e) {
    this.enableZoom && this._handleTouchStartDolly(e), this.enablePan && this._handleTouchStartPan(e);
  }
  _handleTouchStartDollyRotate(e) {
    this.enableZoom && this._handleTouchStartDolly(e), this.enableRotate && this._handleTouchStartRotate(e);
  }
  _handleTouchMoveRotate(e) {
    if (this._pointers.length == 1)
      this._rotateEnd.set(e.pageX, e.pageY);
    else {
      const i = this._getSecondPointerPosition(e), s = 0.5 * (e.pageX + i.x), r = 0.5 * (e.pageY + i.y);
      this._rotateEnd.set(s, r);
    }
    this._rotateDelta.subVectors(this._rotateEnd, this._rotateStart).multiplyScalar(this.rotateSpeed);
    const t = this.domElement;
    this._rotateLeft(qt * this._rotateDelta.x / t.clientHeight), this._rotateUp(qt * this._rotateDelta.y / t.clientHeight), this._rotateStart.copy(this._rotateEnd);
  }
  _handleTouchMovePan(e) {
    if (this._pointers.length === 1)
      this._panEnd.set(e.pageX, e.pageY);
    else {
      const t = this._getSecondPointerPosition(e), i = 0.5 * (e.pageX + t.x), s = 0.5 * (e.pageY + t.y);
      this._panEnd.set(i, s);
    }
    this._panDelta.subVectors(this._panEnd, this._panStart).multiplyScalar(this.panSpeed), this._pan(this._panDelta.x, this._panDelta.y), this._panStart.copy(this._panEnd);
  }
  _handleTouchMoveDolly(e) {
    const t = this._getSecondPointerPosition(e), i = e.pageX - t.x, s = e.pageY - t.y, r = Math.sqrt(i * i + s * s);
    this._dollyEnd.set(0, r), this._dollyDelta.set(0, Math.pow(this._dollyEnd.y / this._dollyStart.y, this.zoomSpeed)), this._dollyOut(this._dollyDelta.y), this._dollyStart.copy(this._dollyEnd);
    const o = (e.pageX + t.x) * 0.5, a = (e.pageY + t.y) * 0.5;
    this._updateZoomParameters(o, a);
  }
  _handleTouchMoveDollyPan(e) {
    this.enableZoom && this._handleTouchMoveDolly(e), this.enablePan && this._handleTouchMovePan(e);
  }
  _handleTouchMoveDollyRotate(e) {
    this.enableZoom && this._handleTouchMoveDolly(e), this.enableRotate && this._handleTouchMoveRotate(e);
  }
  // pointers
  _addPointer(e) {
    this._pointers.push(e.pointerId);
  }
  _removePointer(e) {
    delete this._pointerPositions[e.pointerId];
    for (let t = 0; t < this._pointers.length; t++)
      if (this._pointers[t] == e.pointerId) {
        this._pointers.splice(t, 1);
        return;
      }
  }
  _isTrackingPointer(e) {
    for (let t = 0; t < this._pointers.length; t++)
      if (this._pointers[t] == e.pointerId) return !0;
    return !1;
  }
  _trackPointer(e) {
    let t = this._pointerPositions[e.pointerId];
    t === void 0 && (t = new Ve(), this._pointerPositions[e.pointerId] = t), t.set(e.pageX, e.pageY);
  }
  _getSecondPointerPosition(e) {
    const t = e.pointerId === this._pointers[0] ? this._pointers[1] : this._pointers[0];
    return this._pointerPositions[t];
  }
  //
  _customWheelEvent(e) {
    const t = e.deltaMode, i = {
      clientX: e.clientX,
      clientY: e.clientY,
      deltaY: e.deltaY
    };
    switch (t) {
      case 1:
        i.deltaY *= 16;
        break;
      case 2:
        i.deltaY *= 100;
        break;
    }
    return e.ctrlKey && !this._controlActive && (i.deltaY *= 10), i;
  }
}
function ay(n) {
  this.enabled !== !1 && (this._pointers.length === 0 && (this.domElement.setPointerCapture(n.pointerId), this.domElement.addEventListener("pointermove", this._onPointerMove), this.domElement.addEventListener("pointerup", this._onPointerUp)), !this._isTrackingPointer(n) && (this._addPointer(n), n.pointerType === "touch" ? this._onTouchStart(n) : this._onMouseDown(n)));
}
function ly(n) {
  this.enabled !== !1 && (n.pointerType === "touch" ? this._onTouchMove(n) : this._onMouseMove(n));
}
function cy(n) {
  switch (this._removePointer(n), this._pointers.length) {
    case 0:
      this.domElement.releasePointerCapture(n.pointerId), this.domElement.removeEventListener("pointermove", this._onPointerMove), this.domElement.removeEventListener("pointerup", this._onPointerUp), this.dispatchEvent(Ud), this.state = at.NONE;
      break;
    case 1:
      const e = this._pointers[0], t = this._pointerPositions[e];
      this._onTouchStart({ pointerId: e, pageX: t.x, pageY: t.y });
      break;
  }
}
function uy(n) {
  let e;
  switch (n.button) {
    case 0:
      e = this.mouseButtons.LEFT;
      break;
    case 1:
      e = this.mouseButtons.MIDDLE;
      break;
    case 2:
      e = this.mouseButtons.RIGHT;
      break;
    default:
      e = -1;
  }
  switch (e) {
    case Ds.DOLLY:
      if (this.enableZoom === !1) return;
      this._handleMouseDownDolly(n), this.state = at.DOLLY;
      break;
    case Ds.ROTATE:
      if (n.ctrlKey || n.metaKey || n.shiftKey) {
        if (this.enablePan === !1) return;
        this._handleMouseDownPan(n), this.state = at.PAN;
      } else {
        if (this.enableRotate === !1) return;
        this._handleMouseDownRotate(n), this.state = at.ROTATE;
      }
      break;
    case Ds.PAN:
      if (n.ctrlKey || n.metaKey || n.shiftKey) {
        if (this.enableRotate === !1) return;
        this._handleMouseDownRotate(n), this.state = at.ROTATE;
      } else {
        if (this.enablePan === !1) return;
        this._handleMouseDownPan(n), this.state = at.PAN;
      }
      break;
    default:
      this.state = at.NONE;
  }
  this.state !== at.NONE && this.dispatchEvent(Uc);
}
function hy(n) {
  switch (this.state) {
    case at.ROTATE:
      if (this.enableRotate === !1) return;
      this._handleMouseMoveRotate(n);
      break;
    case at.DOLLY:
      if (this.enableZoom === !1) return;
      this._handleMouseMoveDolly(n);
      break;
    case at.PAN:
      if (this.enablePan === !1) return;
      this._handleMouseMovePan(n);
      break;
  }
}
function fy(n) {
  this.enabled === !1 || this.enableZoom === !1 || this.state !== at.NONE || (n.preventDefault(), this.dispatchEvent(Uc), this._handleMouseWheel(this._customWheelEvent(n)), this.dispatchEvent(Ud));
}
function dy(n) {
  this.enabled !== !1 && this._handleKeyDown(n);
}
function py(n) {
  switch (this._trackPointer(n), this._pointers.length) {
    case 1:
      switch (this.touches.ONE) {
        case ys.ROTATE:
          if (this.enableRotate === !1) return;
          this._handleTouchStartRotate(n), this.state = at.TOUCH_ROTATE;
          break;
        case ys.PAN:
          if (this.enablePan === !1) return;
          this._handleTouchStartPan(n), this.state = at.TOUCH_PAN;
          break;
        default:
          this.state = at.NONE;
      }
      break;
    case 2:
      switch (this.touches.TWO) {
        case ys.DOLLY_PAN:
          if (this.enableZoom === !1 && this.enablePan === !1) return;
          this._handleTouchStartDollyPan(n), this.state = at.TOUCH_DOLLY_PAN;
          break;
        case ys.DOLLY_ROTATE:
          if (this.enableZoom === !1 && this.enableRotate === !1) return;
          this._handleTouchStartDollyRotate(n), this.state = at.TOUCH_DOLLY_ROTATE;
          break;
        default:
          this.state = at.NONE;
      }
      break;
    default:
      this.state = at.NONE;
  }
  this.state !== at.NONE && this.dispatchEvent(Uc);
}
function my(n) {
  switch (this._trackPointer(n), this.state) {
    case at.TOUCH_ROTATE:
      if (this.enableRotate === !1) return;
      this._handleTouchMoveRotate(n), this.update();
      break;
    case at.TOUCH_PAN:
      if (this.enablePan === !1) return;
      this._handleTouchMovePan(n), this.update();
      break;
    case at.TOUCH_DOLLY_PAN:
      if (this.enableZoom === !1 && this.enablePan === !1) return;
      this._handleTouchMoveDollyPan(n), this.update();
      break;
    case at.TOUCH_DOLLY_ROTATE:
      if (this.enableZoom === !1 && this.enableRotate === !1) return;
      this._handleTouchMoveDollyRotate(n), this.update();
      break;
    default:
      this.state = at.NONE;
  }
}
function _y(n) {
  this.enabled !== !1 && n.preventDefault();
}
function gy(n) {
  n.key === "Control" && (this._controlActive = !0, this.domElement.getRootNode().addEventListener("keyup", this._interceptControlUp, { passive: !0, capture: !0 }));
}
function vy(n) {
  n.key === "Control" && (this._controlActive = !1, this.domElement.getRootNode().removeEventListener("keyup", this._interceptControlUp, { passive: !0, capture: !0 }));
}
const nr = Object.freeze({
  dry: [3.8, 4.8, 2.25],
  small_room: [2.35, 3.1, 1.5],
  empty_club: [4.82, 6.2, 2.48],
  medium_room: [6.2, 9.3, 2.9],
  cathedral: [8.2, 12.4, 6.8],
  dual_delay: [5.8, 9.6, 3.2],
  outside: [7.2, 11.5, 0]
}), Nh = Object.freeze({
  standard: Object.freeze({
    fov: 38,
    position: Object.freeze([7.5, 6.4, 8.5]),
    target: Object.freeze([0, 1.1, 0]),
    minDistance: 4.2,
    maxDistance: 18
  }),
  compact: Object.freeze({
    fov: 44,
    position: Object.freeze([10.1, 8.13, 11.42]),
    target: Object.freeze([0, 1.35, 0]),
    minDistance: 4.8,
    maxDistance: 22
  })
});
function xy(n = !1) {
  return n ? Nh.compact : Nh.standard;
}
function My(n) {
  return n.space_mode === "dry" ? nr.dry : n.space_mode === "outside" ? nr.outside : n.space_mode === "sfx" ? nr.dual_delay : nr[n.room_preset] ?? nr.medium_room;
}
class Sy {
  constructor(e, t = {}) {
    this.canvas = e, this.compact = !!t.compact, this.renderer = new sy({
      canvas: e,
      antialias: !0,
      alpha: !1,
      preserveDrawingBuffer: !0
    }), this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)), this.renderer.outputColorSpace = sn, this.renderer.toneMapping = nd, this.renderer.toneMappingExposure = 1.05, this.scene = new r0();
    const i = xy(this.compact);
    this.camera = new rn(i.fov, 1, 0.1, 100), this.camera.position.set(...i.position), this.controls = new oy(this.camera, e), this.controls.enableDamping = !0, this.controls.dampingFactor = 0.06, this.controls.minDistance = i.minDistance, this.controls.maxDistance = i.maxDistance, this.controls.target.set(...i.target), this.root = new Dn(), this.room = new Dn(), this.waveGroup = new Dn(), this.atmosphere = new Dn(), this.root.add(this.room, this.waveGroup, this.atmosphere), this.scene.add(this.root), this.source = this.createSource(), this.listener = this.createListener(), this.root.add(this.source, this.listener), this.hemisphere = new p0(14351338, 1515805, 1.7), this.key = new v0(16777215, 2.6), this.key.position.set(4, 7, 4), this.rim = new _0(7730613, 12, 24), this.rim.position.set(-4, 3, -3), this.scene.add(this.hemisphere, this.key, this.rim), this.clock = new M0(), this.running = !0, this.resizeObserver = new ResizeObserver(() => this.resize()), this.resizeObserver.observe(e.parentElement ?? e), this.resize(), this.animate();
  }
  createSource() {
    const e = new Dn(), t = new vt(
      new Dc(0.2, 3),
      new bo({
        color: 10354640,
        emissive: 3663245,
        emissiveIntensity: 2.4,
        roughness: 0.2
      })
    ), i = new vt(
      new Es(0.34, 24, 24),
      new Rn({
        color: 7536566,
        transparent: !0,
        opacity: 0.09,
        depthWrite: !1
      })
    );
    return e.add(t, i), e;
  }
  createListener() {
    const e = new Dn(), t = new bo({
      color: 16764812,
      emissive: 16747586,
      emissiveIntensity: 1.2
    }), i = new vt(new Es(0.12, 20, 20), t);
    i.position.y = 0.34;
    const s = new vt(new Cc(0.11, 0.26, 6, 12), t), r = new vt(
      new Ts(0.34, 0.018, 10, 48),
      new Rn({ color: 16759411, transparent: !0, opacity: 0.55 })
    );
    return r.rotation.x = Math.PI / 2, r.position.y = 0.02, e.add(i, s, r), e;
  }
  clearGroup(e) {
    for (; e.children.length; )
      e.children.pop().traverse?.((i) => {
        i.geometry?.dispose?.(), Array.isArray(i.material) ? i.material.forEach((s) => s.dispose?.()) : i.material?.dispose?.();
      });
  }
  update(e, t, i) {
    this.state = e, this.metrics = t;
    const [s, r, o] = My(e);
    this.displayDimensions = { width: s, length: r, height: o };
    const [a, l] = i.palette, c = new Xe(a), u = new Xe(l);
    this.clearGroup(this.room), this.clearGroup(this.atmosphere), i.mode === "outside" ? this.buildOutside(s, r, c, u, i.time_of_day) : i.mode === "sfx" ? this.buildDualDelay(s, r, o, c, u) : this.buildRoom(s, r, o, c, u, i.mode === "dry"), this.source.position.set(-s * 0.22, 0.34, -r * 0.16), this.listener.position.set(s * 0.16, 0.2, r * 0.22), this.source.children[0].material.color.set(c), this.source.children[0].material.emissive.set(c), this.source.children[1].material.color.set(c), this.buildWaves(c, s, r, t.visual_amount), this.updateLighting(i, c, u);
  }
  buildRoom(e, t, i, s, r, o) {
    const a = new vt(
      new Hs(e, t, 12, 12),
      new bo({
        color: r,
        roughness: 0.78,
        metalness: 0.08,
        transparent: !0,
        opacity: o ? 0.34 : 0.82
      })
    );
    a.rotation.x = -Math.PI / 2, this.room.add(a);
    const l = new ah(
      Math.max(e, t),
      16,
      s,
      r.clone().offsetHSL(0, 0, 0.13)
    );
    l.material.transparent = !0, l.material.opacity = o ? 0.06 : 0.16, l.position.y = 6e-3, this.room.add(l);
    const c = new Ki(e, i, t), u = new vt(
      c,
      new nh({
        color: s,
        transparent: !0,
        opacity: o ? 0.025 : 0.085,
        side: Wt,
        roughness: 0.24,
        transmission: o ? 0 : 0.18,
        depthWrite: !1
      })
    );
    u.position.y = i / 2;
    const h = new Ed(
      new h0(c),
      new Rc({
        color: s,
        transparent: !0,
        opacity: o ? 0.14 : 0.72
      })
    );
    h.position.y = i / 2, this.room.add(u, h);
  }
  buildOutside(e, t, i, s, r) {
    const o = new vt(
      new Hs(e, t, 20, 20),
      new nh({
        color: s,
        transparent: !0,
        opacity: 0.72,
        roughness: 0.78,
        metalness: 0.04
      })
    );
    o.rotation.x = -Math.PI / 2, this.room.add(o);
    const a = new ah(Math.max(e, t), 20, i, s);
    a.material.transparent = !0, a.material.opacity = r === "night" ? 0.15 : 0.1, a.position.y = 6e-3, this.room.add(a);
    const l = new vt(
      new Ts(Math.max(e, t) * 0.43, 0.012, 8, 96),
      new Rn({ color: i, transparent: !0, opacity: 0.42 })
    );
    l.rotation.x = Math.PI / 2, l.position.y = 0.04, this.room.add(l);
    const c = r === "night" ? 74 : 28, u = new Float32Array(c * 3);
    for (let b = 0; b < c; b += 1)
      u[b * 3] = (Math.random() - 0.5) * e * 1.8, u[b * 3 + 1] = 0.8 + Math.random() * 4.6, u[b * 3 + 2] = (Math.random() - 0.5) * t * 1.2;
    const h = new Nt();
    h.setAttribute("position", new En(u, 3));
    const f = new u0(
      h,
      new Td({
        color: r === "night" ? 13095423 : i,
        size: r === "night" ? 0.035 : 0.022,
        transparent: !0,
        opacity: r === "night" ? 0.78 : 0.32
      })
    );
    this.atmosphere.add(f);
    const p = r === "night", v = p ? 0.22 : 0.5, x = p ? 12174847 : 16760114, m = new vt(
      new Es(v, 24, 24),
      new Rn({
        color: x,
        transparent: !0,
        opacity: p ? 0.88 : 1,
        toneMapped: !1
      })
    ), d = new vt(
      new Es(v * (p ? 1.65 : 2.25), 24, 24),
      new Rn({
        color: x,
        transparent: !0,
        opacity: p ? 0.08 : 0.13,
        depthWrite: !1,
        toneMapped: !1
      })
    );
    m.add(d), m.position.set(-e * 0.38, 3.8, -t * 0.32), this.atmosphere.add(m);
  }
  buildDualDelay(e, t, i, s, r) {
    this.buildRoom(e, t, i, s, r, !1);
    const o = new Rn({
      color: s,
      transparent: !0,
      opacity: 0.64,
      depthWrite: !1
    }), a = new Rn({
      color: s,
      transparent: !0,
      opacity: 0.28,
      depthWrite: !1
    });
    [-1, 1].forEach((l, c) => {
      const u = new vt(
        new Ki(0.035, 0.035, t * 0.72),
        o.clone()
      );
      u.position.set(l * e * 0.2, 0.16 + c * 0.08, 0), this.room.add(u);
      for (let h = 0; h < 5; h += 1) {
        const f = new vt(
          new Ts(0.26 + h * 0.09, 0.012, 8, 48),
          a.clone()
        );
        f.rotation.x = Math.PI / 2, f.position.set(
          l * e * 0.2,
          0.18 + c * 0.08,
          -t * 0.25 + h * t * 0.125
        ), this.room.add(f);
      }
    });
  }
  buildWaves(e, t, i, s) {
    if (this.clearGroup(this.waveGroup), s <= 0) return;
    const r = 3 + Math.round(s * 4);
    for (let o = 0; o < r; o += 1) {
      const a = new vt(
        new Ts(0.35, 0.012, 8, 64),
        new Rn({ color: e, transparent: !0, opacity: 0.18, depthWrite: !1 })
      );
      a.rotation.x = Math.PI / 2, a.position.copy(this.source.position), a.userData = { index: o, count: r, maxScale: Math.max(t, i) * 1.2 }, this.waveGroup.add(a);
    }
  }
  updateLighting(e, t, i) {
    e.mode === "outside" && e.time_of_day === "day" ? (this.scene.background = new Xe(2893592), this.hemisphere.color.set(16773053), this.hemisphere.groundColor.set(4011808), this.key.color.set(16762972), this.key.intensity = 4.1) : e.mode === "outside" ? (this.scene.background = new Xe(329751), this.hemisphere.color.set(9215999), this.hemisphere.groundColor.set(592660), this.key.color.set(8624127), this.key.intensity = 1.3) : (this.scene.background = i.clone().multiplyScalar(0.18), this.hemisphere.color.set(14351338), this.hemisphere.groundColor.set(i), this.key.color.set(16777215), this.key.intensity = e.mode === "dry" ? 1.6 : 2.6), this.rim.color.set(t), this.rim.intensity = e.mode === "dry" ? 4 : 12;
  }
  resize() {
    const e = this.canvas.parentElement, t = Math.max(1, e?.clientWidth ?? this.canvas.clientWidth), i = Math.max(1, e?.clientHeight ?? this.canvas.clientHeight);
    this.renderer.setSize(t, i, !1), this.camera.aspect = t / i, this.camera.updateProjectionMatrix();
  }
  animate = () => {
    if (!this.running) return;
    const e = this.clock.getElapsedTime(), t = this.metrics?.visual_amount ?? 0, i = 0.22 + t * 0.22;
    for (const s of this.waveGroup.children) {
      const r = (e * i + s.userData.index / s.userData.count) % 1, o = 0.2 + r * s.userData.maxScale;
      s.scale.setScalar(o), s.material.opacity = (1 - r) * (0.08 + t * 0.22);
    }
    this.source.rotation.y = e * 0.4, this.atmosphere.rotation.y = Math.sin(e * 0.08) * 0.035, this.controls.update(), this.renderer.render(this.scene, this.camera), this.animationFrame = requestAnimationFrame(this.animate);
  };
  toDataURL() {
    return this.renderer.render(this.scene, this.camera), this.canvas.toDataURL("image/png");
  }
  dispose() {
    this.running = !1, cancelAnimationFrame(this.animationFrame), this.resizeObserver.disconnect(), this.controls.dispose(), this.clearGroup(this.root), this.renderer.dispose();
  }
}
const yy = (n, e) => {
  const t = n.__vccOpts || n;
  for (const [i, s] of e)
    t[i] = s;
  return t;
}, Ey = { class: "akuspace-widget__header" }, Ty = ["aria-expanded", "aria-label"], by = { class: "akuspace-widget__mode-fader" }, Ay = ["max", "value"], wy = {
  key: 0,
  class: "akuspace-widget__faders"
}, Ry = {
  key: 1,
  class: "akuspace-widget__faders"
}, Cy = ["max", "value"], Py = ["max", "value"], Dy = {
  key: 2,
  class: "akuspace-widget__faders"
}, Ly = { class: "akuspace-widget__segments akuspace-widget__segments--two" }, Iy = ["onClick"], Uy = {
  key: 3,
  class: "akuspace-widget__faders"
}, Ny = ["max", "value"], Fy = { class: "akuspace-widget__ticks akuspace-widget__ticks--two" }, Oy = { class: "akuspace-widget__prompt-preview" }, By = {
  __name: "AcousticSpaceWidget",
  props: {
    initialState: { type: Object, default: () => ({}) },
    initialPrompt: { type: String, default: "" },
    previewLabel: { type: String, default: "Prompt output" },
    onStateChange: { type: Function, default: null }
  },
  setup(n, { expose: e }) {
    const t = n, i = /* @__PURE__ */ ns(null), s = /* @__PURE__ */ ns(null), r = /* @__PURE__ */ ns(null), o = /* @__PURE__ */ vr({ ...pi, ...t.initialState }), a = /* @__PURE__ */ ns(t.initialPrompt), l = /* @__PURE__ */ ns(!1), c = /* @__PURE__ */ ns(!1), u = /* @__PURE__ */ vr({ x: 0, y: 0 });
    let h = null, f = null, p = null;
    const v = nn(() => vc(o)), x = nn(() => V_(o)), m = nn(() => Math.max(
      0,
      is.findIndex((Ge) => Ge.value === o.space_mode)
    )), d = nn(() => is[m.value]?.label ?? "Room"), b = nn(() => W_(
      a.value,
      G_(o),
      o.space_mode !== "dry"
    )), A = nn(() => Math.max(0, xs.indexOf(o.room_preset))), M = nn(() => Math.max(0, Ms.indexOf(o.effect_level))), C = nn(() => Math.max(0, Ss.indexOf(o.sfx_level))), w = nn(() => o.sfx_level === "high" ? "High" : "Low"), P = nn(() => l.value || c.value), U = nn(() => ({
      transform: `translate(calc(-50% + ${u.x}px), ${u.y}px)`
    })), S = nn(() => o.space_mode === "dry" ? "Application · Off" : o.space_mode === "outside" ? `Space · ${o.outdoor_time === "night" ? "Night" : "Day"}` : o.space_mode === "sfx" ? `Sound effects · ${w.value}` : `Room · ${dr[o.effect_level]?.label ?? "Moderate"}`);
    function y(Ge) {
      return { gridTemplateColumns: `repeat(${Ge}, 1fr)` };
    }
    function D() {
      h?.update(o, x.value, v.value);
    }
    function L(Ge = {}) {
      Object.assign(o, Ge);
    }
    function V(Ge = "") {
      a.value = Ge;
    }
    function Z() {
      f !== null && window.clearTimeout(f), f = null;
    }
    function ne() {
      Z(), c.value = !0;
    }
    function J() {
      Z(), f = window.setTimeout(() => {
        c.value = !1;
      }, 180);
    }
    function ie() {
      Z(), l.value = !l.value, c.value = l.value;
    }
    function H() {
      Z(), l.value = !1, c.value = !1;
    }
    function fe(Ge, Ae, X) {
      return Math.min(X, Math.max(Ae, Ge));
    }
    function ge(Ge) {
      if (!p || !i.value || !r.value) return;
      const Ae = i.value.getBoundingClientRect(), X = r.value.getBoundingClientRect(), re = Math.max(0, (Ae.width - X.width) / 2 - 8), be = Math.max(0, Ae.height - X.height - 46);
      u.x = fe(p.x + Ge.clientX - p.clientX, -re, re), u.y = fe(p.y + Ge.clientY - p.clientY, 0, be);
    }
    function ye() {
      p = null, window.removeEventListener("pointermove", ge), window.removeEventListener("pointerup", ye), window.removeEventListener("pointercancel", ye);
    }
    function Fe(Ge) {
      Z(), l.value = !0, p = {
        clientX: Ge.clientX,
        clientY: Ge.clientY,
        x: u.x,
        y: u.y
      }, window.addEventListener("pointermove", ge), window.addEventListener("pointerup", ye, { once: !0 }), window.addEventListener("pointercancel", ye, { once: !0 });
    }
    function Je() {
      Z(), ye(), h?.dispose(), h = null;
    }
    return xo(
      o,
      () => {
        D(), t.onStateChange?.({ ...o });
      },
      { deep: !0 }
    ), pc(() => {
      h = new Sy(s.value, { compact: !0 }), D();
    }), mc(Je), e({ setState: L, setPrompt: V, cleanup: Je }), (Ge, Ae) => (Lt(), Ot("div", {
      ref_key: "rootRef",
      ref: i,
      class: "akuspace-widget",
      style: gi({ "--ak-accent": v.value.palette[0] })
    }, [
      Ne("canvas", {
        ref_key: "canvasRef",
        ref: s,
        "aria-label": "Interactive AKUSPACE room preview"
      }, null, 512),
      Ne("div", Ey, [
        Ae[4] || (Ae[4] = Ne("span", null, "AKUSPACE", -1)),
        Ne("button", {
          class: "akuspace-widget__toggle",
          type: "button",
          "aria-expanded": P.value,
          "aria-label": P.value ? "Fold acoustic controls" : "Open acoustic controls",
          onMouseenter: ne,
          onMouseleave: J,
          onClick: ie
        }, [
          Ne("i", {
            class: mr({ open: P.value })
          }, null, 2)
        ], 40, Ty),
        Ne("strong", null, jt(v.value.label), 1)
      ]),
      $t($m, { name: "akuspace-panel" }, {
        default: mf(() => [
          Bp(Ne("div", {
            ref_key: "panelRef",
            ref: r,
            class: "akuspace-widget__controls",
            style: gi(U.value),
            onMouseenter: ne,
            onMouseleave: J
          }, [
            Ne("button", {
              class: "akuspace-widget__dragbar",
              type: "button",
              "aria-label": "Move acoustic controls",
              onPointerdown: y_(Fe, ["prevent"])
            }, [
              Ae[5] || (Ae[5] = Ne("i", null, null, -1)),
              Ne("span", null, jt(S.value), 1),
              Ae[6] || (Ae[6] = Ne("i", null, null, -1))
            ], 32),
            Ne("label", by, [
              Ne("span", null, [
                Ae[7] || (Ae[7] = Ne("span", null, "Mode", -1)),
                Ne("strong", null, jt(d.value), 1)
              ]),
              Ne("input", {
                type: "range",
                min: "0",
                max: yt(is).length - 1,
                step: "1",
                value: m.value,
                "aria-label": "AKUSPACE mode",
                onInput: Ae[0] || (Ae[0] = (X) => L({ space_mode: yt(is)[Number(X.target.value)].value }))
              }, null, 40, Ay),
              Ne("small", {
                class: "akuspace-widget__ticks",
                style: gi(y(yt(is).length))
              }, [
                (Lt(!0), Ot(Vt, null, Ys(yt(is), (X) => (Lt(), Ot("i", {
                  key: X.value
                }, jt(X.value === "sfx" ? "SFX" : X.label), 1))), 128))
              ], 4)
            ]),
            o.space_mode === "dry" ? (Lt(), Ot("div", wy, [...Ae[8] || (Ae[8] = [
              Ne("div", { class: "akuspace-widget__effect" }, [
                Ne("span", null, "AKUSPACE"),
                Ne("small", null, "Off")
              ], -1)
            ])])) : sr("", !0),
            o.space_mode === "room" ? (Lt(), Ot("div", Ry, [
              Ne("label", null, [
                Ne("span", null, [
                  Ae[9] || (Ae[9] = Ne("span", null, "Reverb size", -1)),
                  Ne("strong", null, jt(v.value.short_label), 1)
                ]),
                Ne("input", {
                  type: "range",
                  min: "0",
                  max: yt(xs).length - 1,
                  step: "1",
                  value: A.value,
                  onInput: Ae[1] || (Ae[1] = (X) => L({ room_preset: yt(xs)[Number(X.target.value)] }))
                }, null, 40, Cy),
                Ne("small", {
                  class: "akuspace-widget__ticks",
                  style: gi(y(yt(xs).length))
                }, [
                  (Lt(!0), Ot(Vt, null, Ys(yt(xs), (X) => (Lt(), Ot("i", { key: X }, jt(yt(Jf)[X].short_label), 1))), 128))
                ], 4)
              ]),
              Ne("label", null, [
                Ne("span", null, [
                  Ae[10] || (Ae[10] = Ne("span", null, "Dry / wet", -1)),
                  Ne("strong", null, jt(yt(dr)[o.effect_level]?.label), 1)
                ]),
                Ne("input", {
                  type: "range",
                  min: "0",
                  max: yt(Ms).length - 1,
                  step: "1",
                  value: M.value,
                  onInput: Ae[2] || (Ae[2] = (X) => L({ effect_level: yt(Ms)[Number(X.target.value)] }))
                }, null, 40, Py),
                Ne("small", {
                  class: "akuspace-widget__ticks",
                  style: gi(y(yt(Ms).length))
                }, [
                  (Lt(!0), Ot(Vt, null, Ys(yt(Ms), (X) => (Lt(), Ot("i", { key: X }, jt(yt(dr)[X].label), 1))), 128))
                ], 4)
              ])
            ])) : sr("", !0),
            o.space_mode === "outside" ? (Lt(), Ot("div", Dy, [
              Ne("div", Ly, [
                (Lt(!0), Ot(Vt, null, Ys(yt(O_), (X) => (Lt(), Ot("button", {
                  key: X,
                  class: mr({ active: o.outdoor_time === X }),
                  onClick: (re) => L({ outdoor_time: X })
                }, jt(X === "day" ? "Day" : "Night"), 11, Iy))), 128))
              ])
            ])) : sr("", !0),
            o.space_mode === "sfx" ? (Lt(), Ot("div", Uy, [
              Ae[12] || (Ae[12] = Ne("div", { class: "akuspace-widget__effect" }, [
                Ne("span", null, "Dual Delay"),
                Ne("small", null, "Experimental SFX")
              ], -1)),
              Ne("label", null, [
                Ne("span", null, [
                  Ae[11] || (Ae[11] = Ne("span", null, "Dry / wet", -1)),
                  Ne("strong", null, jt(w.value), 1)
                ]),
                Ne("input", {
                  type: "range",
                  min: "0",
                  max: yt(Ss).length - 1,
                  step: "1",
                  value: C.value,
                  onInput: Ae[3] || (Ae[3] = (X) => L({ sfx_level: yt(Ss)[Number(X.target.value)] }))
                }, null, 40, Ny),
                Ne("small", Fy, [
                  (Lt(!0), Ot(Vt, null, Ys(yt(Ss), (X) => (Lt(), Ot("i", { key: X }, jt(X === "high" ? "High" : "Low"), 1))), 128))
                ])
              ])
            ])) : sr("", !0),
            Ne("div", Oy, [
              Ne("span", null, jt(n.previewLabel), 1),
              Ne("p", null, jt(b.value || "Add visual text in the Text field."), 1)
            ]),
            Ne("button", {
              class: "akuspace-widget__fold",
              type: "button",
              onClick: H
            }, [...Ae[13] || (Ae[13] = [
              Ne("i", null, null, -1),
              Ne("span", null, "Fold controls", -1),
              Ne("i", null, null, -1)
            ])])
          ], 36), [
            [n_, P.value]
          ])
        ]),
        _: 1
      })
    ], 4));
  }
}, zy = /* @__PURE__ */ yy(By, [["__scopeId", "data-v-4698caf0"]]), { app: Nc } = window.comfyAPI.app;
(() => {
  const n = "akuspace-spatial-widget-styles";
  if (document.getElementById(n)) return;
  const e = document.createElement("link");
  e.id = n, e.rel = "stylesheet", e.href = new URL(
    /* @vite-ignore */
    "./assets/acoustic-space-widget.css",
    import.meta.url
  ).href, document.head.appendChild(e);
})();
const As = /* @__PURE__ */ new WeakMap(), Fc = "akuspaceSpatialState", Hy = 200, Nd = [
  "space_mode",
  "application",
  "room_preset",
  "effect_level",
  "outdoor_time",
  "sfx_level"
], Vy = {
  Off: "dry",
  Room: "room",
  Space: "outside",
  "Sound effects": "sfx",
  dry: "dry",
  room: "room",
  outside: "outside",
  sfx: "sfx"
}, ky = {
  dry: "Off",
  room: "Room",
  outside: "Space",
  sfx: "Sound effects"
}, Fh = {
  dry: ["Off"],
  room: ["Low", "Moderate", "Heavy"],
  outside: ["Day", "Night"],
  sfx: ["Low", "High"]
}, Gy = /* @__PURE__ */ new Set([
  "room_preset",
  "effect_level",
  "outdoor_time",
  "sfx_level"
]), Wy = /* @__PURE__ */ new Set([
  "AcousticSpaceReference",
  "AcousticSpaceTextEncode",
  "Koshi_AKUSPACEPrompt",
  "Koshi_AKUSPACETextEncode"
]), Xy = /* @__PURE__ */ new Set([
  "AcousticSpaceTextEncode",
  "Koshi_AKUSPACETextEncode"
]);
function Fd(n) {
  return Vy[n] ?? pi.space_mode;
}
function Od(n) {
  return n.space_mode === "dry" ? "Off" : n.space_mode === "outside" ? n.outdoor_time === "night" ? "Night" : "Day" : n.space_mode === "sfx" ? n.sfx_level === "high" ? "High" : "Low" : n.effect_level === "low" ? "Low" : n.effect_level === "high" ? "Heavy" : "Moderate";
}
function Bd(n, e) {
  return n.space_mode === "outside" && (n.outdoor_time = e === "Night" ? "night" : "day"), n.space_mode === "sfx" && (n.sfx_level = e === "High" ? "high" : "low"), n.space_mode === "room" && (n.effect_level = e === "Low" ? "low" : e === "Heavy" ? "high" : "mid"), n;
}
function _i(n, e, t) {
  return n.widgets?.find((i) => i.name === e)?.value ?? t;
}
function Oh(n) {
  const e = n.properties?.[Fc];
  return e && typeof e == "object" ? e : null;
}
function Oc(n) {
  const e = Oh(n) ?? {}, t = {
    space_mode: Fd(
      e.space_mode ?? _i(n, "space_mode", pi.space_mode)
    ),
    room_preset: e.room_preset ?? _i(n, "room_preset", pi.room_preset),
    effect_level: e.effect_level ?? _i(n, "effect_level", pi.effect_level),
    outdoor_time: e.outdoor_time ?? _i(n, "outdoor_time", pi.outdoor_time),
    sfx_preset: pi.sfx_preset,
    sfx_level: e.sfx_level ?? _i(n, "sfx_level", pi.sfx_level)
  };
  return Oh(n) || Bd(t, _i(n, "application", Od(t))), t;
}
function zd(n, e) {
  n.properties || (n.properties = {}), n.properties[Fc] = { ...e };
}
function Bc(n, e) {
  for (const t of Nd) {
    const i = n.widgets?.find((s) => s.name === t);
    i && (t === "space_mode" ? i.value = ky[e.space_mode] ?? "Room" : t === "application" ? (i.options.values = Fh[e.space_mode] ?? Fh.room, i.value = Od(e)) : e[t] !== void 0 && (i.value = e[t]));
  }
}
function Yy(n) {
  for (const e of n.widgets ?? [])
    !Gy.has(e.name) || e.__akuspaceHidden || (e.__akuspaceHidden = !0, e.__akuspaceOriginalType = e.type, e.__akuspaceOriginalComputeSize = e.computeSize, e.__akuspaceOriginalDraw = e.draw, e.type = "converted-widget", e.computeSize = () => [0, -4], e.draw = () => {
    }, e.hidden = !0, e.options = { ...e.options, hidden: !0 });
}
function qy(n) {
  const e = n.outputs?.[0];
  if (!e) return;
  const t = zc(n) ? "Conditioning" : "Prompt";
  e.name = t, e.label = t, e.localized_name = t;
}
function el(n) {
  Yy(n), qy(n);
  const [e, t] = n.size, i = zc(n) ? 560 : 470;
  n.setSize([Math.max(e, 360), Math.max(t, i)]), Nc.graph?.setDirtyCanvas(!0, !0);
}
function jy(n) {
  const e = document.createElement("div");
  e.className = "akuspace-widget-host", e.style.cssText = "width:100%;height:100%;min-height:300px";
  const t = {
    container: e,
    currentNode: n,
    widget: null,
    cleanupTimer: null,
    vueApp: null,
    exposed: null
  }, i = b_(zy, {
    initialState: Oc(n),
    initialPrompt: _i(n, "text", ""),
    previewLabel: zc(n) ? "Combined prompt" : "Prompt output",
    onStateChange: (s) => {
      const r = t.currentNode;
      Bc(r, s), zd(r, s), Nc.graph?.setDirtyCanvas(!0, !0);
    }
  });
  return t.exposed = i.mount(e), t.vueApp = i, As.set(n, t), t;
}
function Ky(n, e) {
  for (const t of Nd) {
    const i = n.widgets?.find((r) => r.name === t);
    if (!i || i.__akuspaceBound) continue;
    const s = i.callback;
    i.callback = function(...r) {
      s?.apply(this, r);
      const o = Oc(n);
      t === "space_mode" ? o.space_mode = Fd(i.value) : t === "application" ? Bd(o, i.value) : o[t] = i.value, Bc(n, o), zd(n, o), e.exposed.setState(o);
    }, i.__akuspaceBound = !0;
  }
}
function $y(n, e) {
  const t = n.widgets?.find((s) => s.name === "text");
  if (!t || t.__akuspacePromptBound) return;
  const i = t.callback;
  t.callback = function(...s) {
    i?.apply(this, s), e.exposed.setPrompt(t.value ?? "");
  }, t.__akuspacePromptBound = !0;
}
function Zy(n) {
  let e = As.get(n);
  e ? (e.cleanupTimer !== null && clearTimeout(e.cleanupTimer), e.cleanupTimer = null, e.currentNode = n, e.exposed.setState(Oc(n)), e.exposed.setPrompt(_i(n, "text", ""))) : e = jy(n);
  const t = n.addDOMWidget(
    "space_preview",
    "akuspace-spatial-preview",
    e.container,
    { getMinHeight: () => 300, hideOnZoom: !1, serialize: !1 }
  );
  e.widget = t, Ky(n, e), $y(n, e);
  const i = t.onRemove?.bind(t);
  t.onRemove = () => {
    i?.();
    const s = As.get(n);
    !s || s.widget !== t || (s.cleanupTimer = window.setTimeout(() => {
      const r = As.get(n);
      !r || r.widget !== t || (r.exposed.cleanup(), r.vueApp.unmount(), As.delete(n));
    }, Hy));
  };
}
function Jy(n, e) {
  const t = n.onPropertyChanged;
  n.onPropertyChanged = function(i, s) {
    t?.call(this, i, s), !(i !== Fc || !s || typeof s != "object") && (Bc(n, s), e.exposed.setState(s));
  };
}
function Hd(n) {
  return [
    n?.constructor?.comfyClass,
    n?.comfyClass,
    n?.type,
    n?.constructor?.type
  ].find((t) => Wy.has(t));
}
function zc(n) {
  return Xy.has(Hd(n));
}
function Qy(n) {
  return !!Hd(n);
}
Nc.registerExtension({
  name: "Koshi.AKUSPACE.SpatialControl",
  nodeCreated(n) {
    if (!Qy(n)) return;
    el(n), Zy(n);
    const e = As.get(n);
    e && Jy(n, e), window.requestAnimationFrame(() => el(n)), window.setTimeout(() => el(n), 120);
  }
});
